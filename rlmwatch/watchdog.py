"""In-pod watchdog: a supervisor thread inside the training process.

What it has that the sentinel does not: sub-second visibility, stack traces,
`nvidia-smi`, the filesystem, and the ability to request a checkpoint before
anything dies.

What it can never do: **report the pod's own death.** It dies with the pod. Any
design that treats the watchdog as sufficient has an unmonitored failure class
sitting in the middle of it, which is why `rlmwatch.sentinel` exists and why
`WATCHDOG_PROBES` deliberately omits the pod-level checks.

The trainer's side of the contract is three calls:

    watchdog.phase("rollout")       # entering a long phase
    watchdog.heartbeat()            # inside it, periodically
    watchdog.step(n, max_steps=250) # a step completed

`phase` is what makes staleness thresholds meaningful; `heartbeat` is what keeps
a legitimately long rollout from looking like a hang.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from rlmwatch.actions import EscalationLadder
from rlmwatch.config import RunConfig
from rlmwatch.probes import watchdog_probes
from rlmwatch.probes.base import BaseProbe, Context, Verdict

log = logging.getLogger("rlmwatch.watchdog")


class Watchdog:
    """Runs the in-pod probe sweep on a timer, in a daemon thread.

    Daemon on purpose: the watchdog must never be the reason a process refuses
    to exit. Its job is to notice trouble, not to hold the door open.
    """

    def __init__(
        self,
        cfg: RunConfig,
        *,
        ladder: EscalationLadder,
        wandb_client: Any,
        runpod_client: Any,
        probes: tuple[BaseProbe, ...] | None = None,
        interval_s: float = 30.0,
        gen_backend_health: Callable[[], bool] | None = None,
        baselines: dict[str, float] | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.ladder = ladder
        self.probes = probes if probes is not None else watchdog_probes(cfg.run.regime)
        self.interval_s = interval_s
        self.gen_backend_health = gen_backend_health
        self._now = now

        self._ctx = Context(
            cfg=cfg,
            wandb=wandb_client,
            runpod=runpod_client,
            baselines=dict(baselines or {}),
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.ticks = 0
        self.last_verdicts: list[Verdict] = []

    # --- the trainer's side of the contract ----------------------------------

    def phase(self, name: str) -> None:
        """Declare the current training phase.

        Without this every threshold is a compromise between false alarms on
        long rollouts and uselessly long timeouts on short updates.
        """
        with self._lock:
            self._ctx.phase = name
            self._ctx.local["phase_started_at"] = self._now()
            self._ctx.local["heartbeat"] = 0

    def heartbeat(self) -> None:
        """Signal liveness inside a long phase.

        A GRPO rollout can legitimately log nothing for many minutes. This is
        how it stays distinguishable from a wedged one.
        """
        with self._lock:
            self._ctx.local["heartbeat"] = self._ctx.local.get("heartbeat", 0) + 1
            self._ctx.local["step_updated_at"] = self._now()
            self._ctx.local["now"] = self._now()

    def step(self, n: int, *, max_steps: int | None = None, **extra: Any) -> None:
        """Record a completed step."""
        with self._lock:
            self._ctx.local["step"] = n
            self._ctx.local["step_updated_at"] = self._now()
            self._ctx.local["now"] = self._now()
            if max_steps is not None:
                self._ctx.local["max_steps"] = max_steps
            self._ctx.local.update(extra)

    def set_baseline(self, name: str, seconds: float) -> None:
        with self._lock:
            self._ctx.baselines[name] = seconds

    @property
    def context(self) -> Context:
        return self._ctx

    @property
    def halted(self) -> bool:
        """True once the ladder has decided the run should stop.

        The training loop checks this to break cleanly, which gets a checkpoint
        written at a safe point rather than wherever a signal happened to land.
        """
        return self.ladder.halt_requested

    # --- the sweep -----------------------------------------------------------

    def tick(self) -> list[Verdict]:
        """One probe sweep. Returns verdicts; the ladder decides what happens."""
        with self._lock:
            self._ctx.local["now"] = self._now()
            if self.gen_backend_health is not None:
                try:
                    self._ctx.local["gen_backend_healthy"] = self.gen_backend_health()
                except Exception as exc:  # noqa: BLE001 - unreachable is not unhealthy
                    log.warning("generation backend health check raised: %s", exc)
                    self._ctx.local["gen_backend_healthy"] = None
            ctx = self._ctx

        verdicts: list[Verdict] = []
        for probe in self.probes:
            try:
                verdict = probe.check(ctx)
            except Exception as exc:  # noqa: BLE001 - a probe crash blinds, never kills
                verdict = Verdict(
                    probe=probe.name, status="unknown",
                    detail=f"probe raised {type(exc).__name__}: {exc}",
                    evidence={"error": str(exc)},
                )
            verdicts.append(verdict)
            # A confirmed fail needs a second reading from the same probe;
            # re-running it is the cheapest honest re-poll available in-process.
            self.ladder.handle(verdict, recheck=lambda p=probe: p.check(ctx))

        self.ticks += 1
        self.last_verdicts = verdicts
        self.ladder.notifier.heartbeat()
        return verdicts

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - the watchdog outlives its own bugs
                log.exception("watchdog tick failed: %s", exc)

    def start(self) -> Watchdog:
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._loop, name="rlmwatch-watchdog",
                                        daemon=True)
        self._thread.start()
        log.info("watchdog started (%d probes, %.0fs interval)",
                 len(self.probes), self.interval_s)
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # --- context manager -----------------------------------------------------

    def __enter__(self) -> Watchdog:
        return self.start()

    def __exit__(self, exc_type, exc, _tb) -> bool:
        """Guarantee that no exit path leaves a billing pod.

        This is the mandatory guard from the spec, expressed as a context
        manager so the training script cannot forget it. Normal completion,
        exception and SIGTERM all land here; SIGKILL, OOM-kill and host failure
        do not, and that is precisely what the sentinel's dead-man's switch
        covers.
        """
        self.stop()
        if exc_type is None:
            self.ladder.shutdown("completed")
        elif issubclass(exc_type, KeyboardInterrupt):
            self.ladder.shutdown("interrupted")
        else:
            self.ladder.shutdown(f"crashed: {exc!r}")
        return False  # never swallow the exception
