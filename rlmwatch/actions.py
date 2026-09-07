"""The escalation ladder and the mandatory guards (spec section 6).

This is the only module in the library that *does* anything. Probes decide what
is true; this decides what happens about it.

    L0  Log        structured record, no human notified
    L1  Notify     Slack/webhook with evidence, spend, projected spend, W&B link
    L2  Snapshot   py-spy dumps, nvidia-smi, bounded logs, recent metrics
    L3  Halt       cooperative checkpoint, then wandb.finish and stop
    L4  Terminate  wandb.alert -> wandb.finish -> RunPod terminate

Levels are cumulative: escalating to L2 also logs and notifies, because an
operator should never learn about a snapshot without also getting the alert that
prompted it.

Three rules the acceptance criteria turn on:

1. **No probe terminates on a single unconfirmed reading.** Every `fail` is
   re-polled after `confirm_delay_s` before anything past L1 happens.
2. **`unknown` is not `ok`.** Sustained blindness past `unknown_tolerance_s` is
   promoted to `warn` -- a monitor that cannot see is its own incident -- but it
   never becomes a `fail`, so a network partition cannot kill a healthy pod.
3. **Order matters on the way down.** Checkpoint, then alert, then flush metrics,
   and only then terminate. Reverse any two and you lose the evidence or the
   weights.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol

from rlmwatch.config import RunConfig
from rlmwatch.diagnostics import Diagnostics, Snapshot
from rlmwatch.notify import Notifier
from rlmwatch.probes.base import Status, Verdict

log = logging.getLogger("rlmwatch.actions")


class Level(IntEnum):
    LOG = 0
    NOTIFY = 1
    SNAPSHOT = 2
    HALT = 3
    TERMINATE = 4

    @property
    def label(self) -> str:
        return f"L{int(self)}"


#: How far a confirmed `fail` from each probe is allowed to escalate.
#: Everything not listed defaults to `DEFAULT_FAIL_LEVEL`.
#:
#: The split is between "this run is producing nothing of value" (terminate --
#: every further step is spend with no return) and "this needs a human"
#: (snapshot and halt, because the right response depends on judgement the
#: library does not have).
DEFAULT_PROBE_LEVELS: dict[str, Level] = {
    # Budget and wall clock are unconditional caps.
    "cost.spend": Level.TERMINATE,
    "cost.wall_clock": Level.TERMINATE,
    # Nothing is training and nothing will start.
    "liveness.observer_agreement": Level.TERMINATE,
    "liveness.pod_status": Level.NOTIFY,  # the pod is already gone; nothing to kill
    "liveness.progress_age": Level.TERMINATE,
    "liveness.step_counter": Level.TERMINATE,
    "liveness.gpu_util": Level.TERMINATE,
    "liveness.run_state": Level.SNAPSHOT,
    # Training is a no-op or actively destroying the policy.
    "rlm.reward_std": Level.HALT,
    "rlm.kl_ref": Level.HALT,
    "rlm.entropy": Level.HALT,
    "rlm.format_success": Level.HALT,
    "rlm.rollout_duration": Level.HALT,
    "rlm.generation_backend": Level.HALT,
    "rlm.reward_model": Level.HALT,
    "rlm.repl_degeneracy": Level.HALT,
    "health.loss_finite": Level.HALT,
}

DEFAULT_FAIL_LEVEL = Level.SNAPSHOT


class TrainerHooks(Protocol):
    """What L3 needs from a cooperative trainer.

    `request_checkpoint` must return promptly; the trainer saves at its next
    safe point. `wait_for_checkpoint` is bounded, and the ladder escalates past
    L3 when it expires -- a trainer too wedged to checkpoint is exactly the case
    where waiting indefinitely costs the most.
    """

    def request_checkpoint(self) -> None: ...

    def wait_for_checkpoint(self, timeout_s: float) -> bool: ...


@dataclass
class ActionRecord:
    """What the ladder actually did, for the audit trail and for tests."""

    verdict: Verdict
    level: Level
    confirmed: bool
    steps: list[str] = field(default_factory=list)
    snapshot: Snapshot | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe": self.verdict.probe,
            "status": self.verdict.status,
            "level": self.level.label,
            "confirmed": self.confirmed,
            "steps": self.steps,
            "detail": self.verdict.detail,
        }


@dataclass
class _ProbeState:
    """Per-probe memory across ticks."""

    unknown_since: float | None = None
    consecutive_fails: int = 0
    highest_level: Level = Level.LOG


class EscalationLadder:
    """Turns verdicts into actions, once, in order, with confirmation."""

    def __init__(
        self,
        cfg: RunConfig,
        *,
        notifier: Notifier,
        diagnostics: Diagnostics | None = None,
        runpod: Any = None,
        wandb_module: Any = None,
        trainer: TrainerHooks | None = None,
        probe_levels: dict[str, Level] | None = None,
        snapshot_dir: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.notifier = notifier
        self.diagnostics = diagnostics
        self.runpod = runpod
        self.wandb_module = wandb_module
        self.trainer = trainer
        self.probe_levels = {**DEFAULT_PROBE_LEVELS, **(probe_levels or {})}
        self.snapshot_dir = snapshot_dir
        self._sleep = sleep
        self._now = now
        self._state: dict[str, _ProbeState] = {}
        self.history: list[ActionRecord] = []
        self.terminated = False
        #: Set once the ladder has decided the run must stop. The watchdog reads
        #: this to break out of its loop.
        self.halt_requested = False

    # --- policy --------------------------------------------------------------

    def target_level(self, verdict: Verdict) -> Level:
        if verdict.status == "fail":
            return self.probe_levels.get(verdict.probe, DEFAULT_FAIL_LEVEL)
        if verdict.status == "warn":
            return Level.NOTIFY
        return Level.LOG

    def resolve_status(self, verdict: Verdict) -> Status:
        """Promote sustained `unknown` to `warn`; leave everything else alone.

        A monitor that has gone blind is an incident in its own right, but it is
        never evidence that the *run* is broken -- so blindness can reach L1 and
        no further.
        """
        state = self._state.setdefault(verdict.probe, _ProbeState())
        if verdict.status != "unknown":
            state.unknown_since = None
            return verdict.status

        now = self._now()
        if state.unknown_since is None:
            state.unknown_since = now
            return "unknown"
        if now - state.unknown_since >= self.cfg.failsafe.unknown_tolerance_s:
            return "warn"
        return "unknown"

    # --- execution -----------------------------------------------------------

    def _decide(self, verdict: Verdict) -> tuple[Verdict, Level]:
        """Resolve status and target level, without acting or waiting."""
        effective = self.resolve_status(verdict)
        if effective in ("ok", "unknown"):
            return verdict, Level.LOG

        promoted = verdict
        if effective != verdict.status:
            promoted = Verdict(
                probe=verdict.probe,
                status=effective,
                detail=(
                    f"{verdict.detail} (monitor has been blind for "
                    f"{self.cfg.failsafe.unknown_tolerance_s:.0f}s; a monitor that cannot "
                    f"see is itself an incident)"
                ),
                evidence={**verdict.evidence, "promoted_from": verdict.status},
                at=verdict.at,
            )
        return promoted, self.target_level(promoted)

    @staticmethod
    def _downgrade_unconfirmed(verdict: Verdict) -> Verdict:
        return Verdict(
            probe=verdict.probe,
            status="warn",
            detail=f"{verdict.detail} -- did not reproduce on re-poll, not escalating",
            evidence={**verdict.evidence, "confirmed": False},
            at=verdict.at,
        )

    def _finish(self, promoted: Verdict, target: Level, confirmed: bool, *,
                spend_usd: float | None, projected_usd: float | None) -> ActionRecord:
        state = self._state.setdefault(promoted.probe, _ProbeState())
        if target == Level.LOG:
            state.consecutive_fails = 0
            state.highest_level = Level.LOG
            record = ActionRecord(promoted, Level.LOG, confirmed=False, steps=["log"])
            self._log(promoted)
            self.history.append(record)
            return record

        if promoted.status == "fail":
            state.consecutive_fails += 1

        record = ActionRecord(promoted, target, confirmed=confirmed)
        self._execute(record, spend_usd=spend_usd, projected_usd=projected_usd)
        state.highest_level = max(state.highest_level, target)
        self.history.append(record)
        return record

    def handle(
        self,
        verdict: Verdict,
        *,
        recheck: Callable[[], Verdict] | None = None,
        spend_usd: float | None = None,
        projected_usd: float | None = None,
    ) -> ActionRecord:
        """Act on one verdict, paying the confirmation delay if it escalates.

        For a whole sweep prefer `handle_batch`, which pays that delay once
        rather than once per failing probe.
        """
        promoted, target = self._decide(verdict)
        confirmed = False

        # Rule 1: nothing past L1 happens on a single unconfirmed reading.
        if target > Level.NOTIFY:
            confirmed = self._confirm(recheck)
            if not confirmed:
                target = Level.NOTIFY
                promoted = self._downgrade_unconfirmed(promoted)

        return self._finish(promoted, target, confirmed,
                            spend_usd=spend_usd, projected_usd=projected_usd)

    def handle_batch(
        self,
        verdicts: list[Verdict],
        *,
        recheck: Callable[[Verdict], Verdict] | None = None,
        spend_usd: float | None = None,
        projected_usd: float | None = None,
    ) -> list[ActionRecord]:
        """Act on a whole sweep, paying `confirm_delay_s` **once**.

        Confirming serially would make the time-to-action scale with the number
        of failing probes, and probes fail in clusters: a dead trainer trips
        progress age, run state and observer agreement at once. At a 120s delay
        that is six minutes to react to a pod that is already burning money for
        nothing, and it is why the kill-9 acceptance case has a three-minute
        budget. The whole sweep is re-polled together after a single wait.
        """
        decided = [self._decide(v) for v in verdicts]
        needs_confirmation = [
            (i, promoted) for i, (promoted, target) in enumerate(decided)
            if target > Level.NOTIFY
        ]

        confirmations: dict[int, bool] = {}
        if needs_confirmation and recheck is not None:
            self._sleep(self.cfg.failsafe.confirm_delay_s)
            for index, promoted in needs_confirmation:
                try:
                    confirmations[index] = recheck(promoted).status == "fail"
                except Exception as exc:  # noqa: BLE001 - a failed re-poll is not confirmation
                    log.warning("confirmation re-poll for %s failed: %s",
                                promoted.probe, exc)
                    confirmations[index] = False

        records: list[ActionRecord] = []
        for index, (promoted, target) in enumerate(decided):
            confirmed = confirmations.get(index, False)
            if target > Level.NOTIFY and not confirmed:
                target = Level.NOTIFY
                promoted = self._downgrade_unconfirmed(promoted)
            records.append(
                self._finish(promoted, target, confirmed,
                             spend_usd=spend_usd, projected_usd=projected_usd)
            )
        return records

    def _confirm(self, recheck: Callable[[], Verdict] | None) -> bool:
        """Re-poll after `confirm_delay_s`. No recheck available means unconfirmed.

        Refusing to escalate without a second opinion is what keeps a network
        blip from terminating a healthy $30/hr pod.
        """
        if recheck is None:
            return False
        self._sleep(self.cfg.failsafe.confirm_delay_s)
        try:
            second = recheck()
        except Exception as exc:  # noqa: BLE001 - a failed re-poll is not confirmation
            log.warning("confirmation re-poll failed: %s", exc)
            return False
        return second.status == "fail"

    def _execute(self, record: ActionRecord, *, spend_usd: float | None,
                 projected_usd: float | None) -> None:
        """Run every level up to the target, in order."""
        verdict, target = record.verdict, record.level

        self._log(verdict)
        record.steps.append("log")

        if target >= Level.NOTIFY:
            delivered = self.notifier.notify(
                verdict, target.label, spend_usd=spend_usd, projected_usd=projected_usd
            )
            record.steps.append(f"notify({delivered} sink(s))")

        if target >= Level.SNAPSHOT:
            record.snapshot = self._snapshot(verdict)
            record.steps.append(
                f"snapshot({record.snapshot.summary()})" if record.snapshot
                else "snapshot(unavailable)"
            )

        if target >= Level.HALT:
            self.halt_requested = True
            saved = self._checkpoint_and_halt()
            record.steps.append(f"halt(checkpoint_saved={saved})")
            if not saved:
                # A trainer too wedged to checkpoint is precisely the case where
                # waiting longer costs the most. Escalate rather than hang.
                target = Level.TERMINATE
                record.level = target
                record.steps.append("halt_timeout -> escalating to L4")

        if target >= Level.TERMINATE:
            self._terminate(verdict)
            record.steps.append(f"terminate({self.cfg.failsafe.on_terminal})")

    def _log(self, verdict: Verdict) -> None:
        level = {"ok": logging.INFO, "unknown": logging.INFO,
                 "warn": logging.WARNING, "fail": logging.ERROR}[verdict.status]
        log.log(level, "%s | evidence=%s", verdict, verdict.evidence)

    def _snapshot(self, verdict: Verdict) -> Snapshot | None:
        if self.diagnostics is None:
            return None
        snap = self.diagnostics.snapshot(reason=f"{verdict.probe}: {verdict.detail}")
        if self.snapshot_dir:
            try:
                snap.write(self.snapshot_dir)
            except OSError as exc:
                snap.errors["write"] = str(exc)
        self.diagnostics.upload(snap, wandb_module=self.wandb_module)
        return snap

    def _checkpoint_and_halt(self) -> bool:
        """L3. Returns whether a checkpoint was actually saved."""
        if self.trainer is None:
            return False
        try:
            self.trainer.request_checkpoint()
            return self.trainer.wait_for_checkpoint(self.cfg.failsafe.checkpoint_timeout_s)
        except Exception as exc:  # noqa: BLE001
            log.error("checkpoint request failed: %s", exc)
            return False

    def _terminate(self, verdict: Verdict, action: str | None = None) -> None:
        """L4, in the order the spec mandates: alert, flush, then kill.

        `wandb.finish()` before termination is what makes the metrics survive.
        Terminating first leaves the last minutes of the run -- the interesting
        ones -- unflushed and gone.

        `action` overrides `failsafe.on_terminal` for this call only. The
        startup gate uses it: `startup.on_failure` is a separate setting because
        a gate failure happens before the first step, so there are no
        checkpoints to lose and terminate is unambiguously the right answer even
        when the steady-state policy is `stop`. Passing it explicitly is what
        keeps that decision out of the shared config object.
        """
        if self.wandb_module is not None:
            try:
                alert = getattr(self.wandb_module, "alert", None)
                if alert:
                    alert(title=f"rlmwatch: {verdict.probe}", text=verdict.detail)
            except Exception as exc:  # noqa: BLE001
                log.warning("wandb.alert failed: %s", exc)
            try:
                finish = getattr(self.wandb_module, "finish", None)
                if finish:
                    finish()
            except Exception as exc:  # noqa: BLE001
                log.warning("wandb.finish failed: %s", exc)

        action = action or self.cfg.failsafe.on_terminal
        pod_id = self.cfg.run.pod_id
        if not self.runpod or not pod_id:
            log.error(
                "TERMINATION REQUESTED but no RunPod client or pod_id is configured. "
                "The pod is still billing and must be stopped by hand."
            )
            return
        try:
            if action == "terminate":
                self.runpod.terminate(pod_id)
            else:
                self.runpod.stop(pod_id)
            self.terminated = True
        except Exception as exc:  # noqa: BLE001
            log.critical(
                "FAILED TO %s POD %s: %s -- the pod is still billing, stop it by hand",
                action.upper(), pod_id, exc,
            )

    # --- guards --------------------------------------------------------------

    def shutdown(self, reason: str, *, action: str | None = None) -> None:
        """The mandatory exit guard. Wrap training so no exit path leaves a
        billing pod.

        A clean exit does **not** stop billing: the container restarts when the
        entrypoint ends and the meter keeps running. Termination has to be
        explicit, on every path -- normal completion, exception, and SIGTERM
        alike.

        Order: request checkpoint -> wandb.alert -> wandb.finish -> terminate.
        """
        action = action or self.cfg.failsafe.on_terminal
        verdict = Verdict(
            probe="failsafe.shutdown",
            status="fail" if "crash" in reason.lower() else "ok",
            detail=f"shutting down: {reason}",
            evidence={"reason": reason, "action": action},
        )
        log.warning("shutdown(%s) -> %s", reason, action)
        if self.trainer is not None:
            self._checkpoint_and_halt()
        self.notifier.notify(verdict, Level.TERMINATE.label)
        self._terminate(verdict, action)


def install_signal_handlers(ladder: EscalationLadder, *, signal_module=None) -> None:
    """Route SIGTERM/SIGINT through `shutdown` so the pod is never orphaned.

    This does not cover SIGKILL, OOM-kill or host failure -- none of them run
    any handler. That gap is exactly what the sentinel's dead-man's switch is
    for; without it, every failsafe here is best-effort.
    """
    import signal as signal_module_default  # noqa: PLC0415

    sig = signal_module or signal_module_default

    def handler(signum, _frame):  # pragma: no cover - exercised via direct call
        ladder.shutdown(f"signal {signum}")
        raise SystemExit(128 + signum)

    for name in ("SIGTERM", "SIGINT"):
        signum = getattr(sig, name, None)
        if signum is not None:
            sig.signal(signum, handler)
