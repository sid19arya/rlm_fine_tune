"""External sentinel: the observer that survives everything.

It runs on a laptop, a cron job or a cheap VM -- anywhere that is **not the
pod**. It is coarse (minutes, not seconds) and cannot see inside the training
process. In exchange it is the only component that can catch a hung or dead pod,
because it is the only one that outlives one.

Three responsibilities the watchdog structurally cannot have:

1. **The startup deadline.** Absence of `startup_ok` within
   `startup.deadline_s` of pod creation is a failure. This is the only way to
   catch a crash that happens before `wandb.init()`, which W&B can never see
   because the run does not exist yet.
2. **The dead-man's switch.** If nothing has been heard for
   `dead_mans_timeout_s`, terminate. This is the backstop for SIGKILL, OOM-kill
   and host failure -- none of which run anybody's `finally` block. Without it
   every other failsafe is best-effort.
3. **The budget and wall-clock caps.** Unconditional, and enforced from outside
   so they survive the pod becoming unresponsive, which is exactly the state in
   which they matter most.

Cost discipline is a design constraint, not an afterthought: the whole sentinel
has to stay under $1/day including API calls, which is why the default poll is
60s and the W&B client caches within a tick.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rlmwatch.actions import EscalationLadder, Level
from rlmwatch.config import RunConfig
from rlmwatch.probes import sentinel_probes
from rlmwatch.probes.base import BaseProbe, Context, Verdict
from rlmwatch.probes.cost import elapsed_billed_hours

log = logging.getLogger("rlmwatch.sentinel")


@dataclass
class TickResult:
    """One sweep. Serialisable so cron mode can hand it to a human or a log."""

    verdicts: list[Verdict] = field(default_factory=list)
    spend_usd: float | None = None
    projected_usd: float | None = None
    terminated: bool = False

    @property
    def worst_status(self) -> str:
        from rlmwatch.probes.base import worst

        return worst([v.status for v in self.verdicts])

    def as_dict(self) -> dict[str, Any]:
        return {
            "worst": self.worst_status,
            "spend_usd": self.spend_usd,
            "projected_usd": self.projected_usd,
            "terminated": self.terminated,
            "verdicts": [v.as_dict() for v in self.verdicts],
        }


class Sentinel:
    """Polls RunPod and W&B from outside the pod and drives the ladder."""

    def __init__(
        self,
        cfg: RunConfig,
        *,
        ladder: EscalationLadder,
        wandb_client: Any,
        runpod_client: Any,
        probes: tuple[BaseProbe, ...] | None = None,
        pod_created_at: float | None = None,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.ladder = ladder
        self.probes = probes if probes is not None else sentinel_probes(cfg.run.regime)
        self._now = now
        self._sleep = sleep
        self.pod_created_at = pod_created_at if pod_created_at is not None else now()
        self.started_at = self.pod_created_at

        self._ctx = Context(cfg=cfg, wandb=wandb_client, runpod=runpod_client)
        #: Wall-clock of the last tick in which *anything* was readable. The
        #: dead-man's switch measures against this, not against the last
        #: healthy verdict: a monitor that can see a sick run is still seeing.
        self.last_contact = self.pod_created_at
        self.startup_ok_seen = False
        self.ticks = 0
        self.history: list[TickResult] = []

    # --- the three responsibilities only the sentinel can hold ----------------

    def check_startup_deadline(self) -> Verdict | None:
        """Fail if `startup_ok` has not appeared within the deadline.

        Covers the window W&B is blind to: a crash before `wandb.init()` leaves
        no run to report `Crashed`, so the only evidence is the absence of a
        signal that should have arrived.
        """
        if self.startup_ok_seen:
            return None
        age = self._now() - self.pod_created_at
        deadline = self.cfg.startup.deadline_s

        try:
            summary = self._ctx.wandb.summary(self.cfg.run.wandb)
        except Exception:  # noqa: BLE001 - unreadable is not proof of failure
            summary = {}
        if summary.get("startup_ok"):
            self.startup_ok_seen = True
            log.info("startup_ok observed %.0fs after pod creation", age)
            return None

        if age <= deadline:
            return None
        return Verdict(
            probe="sentinel.startup_deadline",
            status="fail",
            detail=(
                f"no startup_ok {age:.0f}s after pod creation (deadline {deadline:.0f}s). "
                f"The run likely died before wandb.init(), which W&B cannot report "
                f"because the run never existed."
            ),
            evidence={"age_s": round(age, 1), "deadline_s": deadline,
                      "pod_id": self.cfg.run.pod_id},
        )

    def check_dead_mans_switch(self) -> Verdict | None:
        """Terminate if nothing has been heard for `dead_mans_timeout_s`.

        The backstop for SIGKILL, OOM-kill and host failure, none of which run
        any handler anywhere. Without it, every other failsafe is best-effort.
        """
        silence = self._now() - self.last_contact
        timeout = self.cfg.failsafe.dead_mans_timeout_s
        if silence < timeout:
            return None
        return Verdict(
            probe="sentinel.dead_mans_switch",
            status="fail",
            detail=(
                f"nothing readable from either observer for {silence:.0f}s "
                f"(timeout {timeout:.0f}s). Neither RunPod nor W&B has answered, so the "
                f"pod is presumed lost and is still billing."
            ),
            evidence={"silence_s": round(silence, 1), "timeout_s": timeout},
        )

    # --- sweep ---------------------------------------------------------------

    def _money(self) -> tuple[float | None, float | None]:
        """Current and projected spend, for every alert this tick emits."""
        hours = elapsed_billed_hours(self._ctx)
        if hours is None:
            return None, None
        spend = self.cfg.budget.spend_at(hours)
        projected = self.cfg.budget.spend_at(self.cfg.budget.max_wall_clock_h)
        return spend, projected

    def _contact(self) -> bool:
        """Did either upstream actually answer this tick?

        Deliberately not inferred from the spend estimate: `elapsed_billed_hours`
        falls back to the monitor's own clock when RunPod is unreachable, so a
        spend number can be produced while nothing has been heard from anything.
        Treating that as contact would disarm the dead-man's switch in exactly
        the situation it exists for.
        """
        reached = False
        try:
            self._ctx.runpod.status(self.cfg.run.pod_id)
            reached = True
        except Exception:  # noqa: BLE001 - unreachable is the thing being measured
            pass
        try:
            self._ctx.wandb.state(self.cfg.run.wandb)
            reached = True
        except Exception:  # noqa: BLE001
            pass
        return reached

    def tick(self) -> TickResult:
        """One full sweep: deadline, probes, dead-man's switch."""
        result = TickResult()
        spend, projected = self._money()
        result.spend_usd, result.projected_usd = spend, projected
        contacted = self._contact()

        deadline_verdict = self.check_startup_deadline()
        if deadline_verdict is not None:
            result.verdicts.append(deadline_verdict)
            self.ladder.handle(deadline_verdict, recheck=self.check_startup_deadline,
                               spend_usd=spend, projected_usd=projected)
            # A pod that never started is not going to start. Terminate rather
            # than continue polling something that will only accrue cost.
            self.ladder.shutdown("startup deadline exceeded")
            result.terminated = True
            self.ticks += 1
            self.history.append(result)
            return result

        by_name = {p.name: p for p in self.probes}
        for probe in self.probes:
            try:
                verdict = probe.check(self._ctx)
            except Exception as exc:  # noqa: BLE001 - a probe crash blinds, never kills
                verdict = Verdict(
                    probe=probe.name, status="unknown",
                    detail=f"probe raised {type(exc).__name__}: {exc}",
                    evidence={"error": str(exc)},
                )
            result.verdicts.append(verdict)

        def recheck(verdict: Verdict) -> Verdict:
            self._ctx.wandb.invalidate(self.cfg.run.wandb)
            probe = by_name.get(verdict.probe)
            if probe is None:  # a promoted verdict with no probe cannot confirm
                return Verdict(probe=verdict.probe, status="unknown",
                               detail="no probe to re-poll")
            return probe.check(self._ctx)

        # One confirmation wait for the whole sweep, not one per failing probe:
        # probes fail in clusters, and serial confirmation would make the
        # time-to-action scale with how badly the run is broken.
        self.ladder.handle_batch(result.verdicts, recheck=recheck, spend_usd=spend,
                                 projected_usd=projected)

        if contacted:
            self.last_contact = self._now()
        else:
            dead_man = self.check_dead_mans_switch()
            if dead_man is not None:
                result.verdicts.append(dead_man)
                # No confirmation re-poll here, and that is deliberate: the
                # switch has already been waiting `dead_mans_timeout_s`, which
                # is far longer than any confirm delay. Requiring another
                # opinion from an observer that has answered nothing for half an
                # hour would only add cost to a pod already presumed lost.
                self.ladder.handle(dead_man, spend_usd=spend, projected_usd=projected)
                self.ladder.shutdown("dead-man's switch")
                result.terminated = True

        self.ticks += 1
        self.history.append(result)
        self.ladder.notifier.heartbeat(fail=result.worst_status == "fail")
        return result

    def run(self, *, max_ticks: int | None = None) -> list[TickResult]:
        """Poll until the run ends, the pod is terminated, or `max_ticks`.

        `max_ticks` is what makes cron mode possible: one tick per invocation,
        no long-lived process to supervise.
        """
        results: list[TickResult] = []
        while max_ticks is None or len(results) < max_ticks:
            result = self.tick()
            results.append(result)
            if result.terminated or self.ladder.terminated:
                log.info("sentinel stopping: pod terminated")
                break
            if self._finished():
                log.info("sentinel stopping: run finished and pod is not running")
                break
            if max_ticks is None or len(results) < max_ticks:
                self._sleep(self.cfg.poll_interval_s)
        return results

    def _finished(self) -> bool:
        """Stop polling only when the run is done *and* the pod is gone.

        A finished run with a live pod is not a reason to stop watching -- it is
        the reason to keep watching, because the container restarts when the
        entrypoint ends and the meter keeps running.
        """
        try:
            state = self._ctx.wandb.state(self.cfg.run.wandb)
        except Exception:  # noqa: BLE001
            return False
        if state not in ("finished", "crashed", "failed", "killed"):
            return False
        try:
            return not self._ctx.runpod.status(self.cfg.run.pod_id).is_running
        except Exception:  # noqa: BLE001
            return False


def build_sentinel(
    cfg: RunConfig,
    *,
    runpod_api_key: str,
    wandb_api_key: str | None = None,
    pod_created_at: float | None = None,
    dry_run: bool = False,
) -> Sentinel:
    """Assemble a sentinel from a config and two credentials.

    `dry_run` caps every probe at L1 so the ladder alerts but never stops or
    terminates anything. It is how you earn confidence in a new config before
    letting it hold the kill switch on a live run.
    """
    from rlmwatch.clients.runpod import RunPodClient
    from rlmwatch.clients.wandb import WandbClient
    from rlmwatch.diagnostics import Diagnostics
    from rlmwatch.notify import Notifier

    runpod = RunPodClient(runpod_api_key)
    wandb_client = WandbClient(wandb_api_key)
    notifier = Notifier(cfg)
    probe_levels = None
    if dry_run:
        probe_levels = {p.name: Level.NOTIFY for p in sentinel_probes(cfg.run.regime)}

    ladder = EscalationLadder(
        cfg,
        notifier=notifier,
        diagnostics=Diagnostics(cfg, runpod=runpod, wandb=wandb_client),
        runpod=None if dry_run else runpod,
        probe_levels=probe_levels,
    )
    return Sentinel(cfg, ladder=ladder, wandb_client=wandb_client, runpod_client=runpod,
                    pod_created_at=pod_created_at)
