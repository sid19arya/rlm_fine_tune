"""Attaching rlmwatch to a training script.

The spec's constraint on this layer is deliberate and worth restating: **add
three lines to `train.py` -- gate, watchdog, shutdown wrapper. Nothing else.**
Anything experiment-specific belongs in the YAML, not in the training script,
because monitoring glue written inside a trainer is glue that gets copied,
diverges, and is never tested.

So the whole integration is one context manager:

    import rlmwatch

    with rlmwatch.attach("configs/rlm-ft-v0-smoke.yaml", hooks=hooks) as watch:
        for step in range(max_steps):
            watch.phase("rollout")
            rollouts = generate(...)          # watch.heartbeat() inside long loops
            watch.phase("update")
            loss = train_on(rollouts)
            watch.step(step, max_steps=max_steps)
            if watch.halted:                  # the ladder asked for a clean stop
                break

Entering runs the startup gate and starts the watchdog. Leaving -- normally, by
exception, or by SIGTERM -- runs the shutdown guard, because a clean exit does
not stop billing: the container restarts when the entrypoint ends and the meter
keeps running.

What this does **not** do is run the sentinel. The sentinel has to live outside
the pod or it dies with the thing it is watching, so it is started separately:

    rlmwatch watch -c configs/rlm-ft-v0-smoke.yaml
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rlmwatch.actions import EscalationLadder, TrainerHooks, install_signal_handlers
from rlmwatch.clients.runpod import RunPodClient
from rlmwatch.clients.wandb import WandbClient
from rlmwatch.config import RunConfig, load_config
from rlmwatch.diagnostics import Diagnostics
from rlmwatch.notify import Notifier
from rlmwatch.probes.base import Context
from rlmwatch.probes.startup import (
    StartupContext,
    StartupHooks,
    SystemHardware,
    gate_and_enforce,
)
from rlmwatch.watchdog import Watchdog

log = logging.getLogger("rlmwatch.integration")


class StartupGateFailed(RuntimeError):
    """The gate failed and the pod has already been stopped or terminated.

    Raised so the training script exits immediately rather than proceeding on
    hardware that is about to disappear.
    """


@contextmanager
def attach(
    config: str | Path | RunConfig,
    *,
    hooks: StartupHooks | None = None,
    trainer: TrainerHooks | None = None,
    gen_backend_health: Callable[[], bool] | None = None,
    wandb_module: Any = None,
    interval_s: float = 30.0,
    skip_gate: bool = False,
    dry_run: bool = False,
) -> Iterator[Watchdog]:
    """Run the startup gate, supervise the run, and guarantee shutdown.

    Args:
        config: path to the run YAML, or an already-loaded RunConfig.
        hooks: `StartupHooks` lending the gate a model load, a first batch, a
            warm-up step and a rollout cycle. Anything not supplied is reported
            as unwired rather than counted as passing -- and the rollout hook is
            what turns `stall_threshold: auto` into a measured number instead of
            the fallback floor.
        trainer: cooperative checkpoint hooks, so L3 can save before halting.
            Without it the ladder cannot checkpoint and escalates straight to
            termination.
        gen_backend_health: called each watchdog tick. Only reachable from
            inside the pod, which is why the watchdog owns it.
        skip_gate: for local dry runs. Never in the pod -- the gate is what
            turns a six-hour failure into a ninety-second one.
        dry_run: no RunPod client is constructed at all, so nothing can be
            stopped or terminated and no credential is required. Use it to
            check the wiring on a laptop before it holds the kill switch on a
            live pod. Never set it in the pod: it disables every failsafe.

    Yields the `Watchdog`, whose `phase`, `heartbeat` and `step` are the
    trainer's side of the contract.
    """
    cfg = config if isinstance(config, RunConfig) else load_config(config)

    if wandb_module is None:
        try:
            import wandb as wandb_module  # noqa: PLC0415 - optional in tests
        except ImportError:
            wandb_module = None

    runpod: RunPodClient | None = None
    if dry_run:
        log.warning(
            "DRY RUN: no RunPod client. Nothing can be stopped or terminated, so "
            "every failsafe is inert. Never use this inside a pod."
        )
    else:
        runpod_key = os.environ.get("RUNPOD_API_KEY", "")
        if not runpod_key:
            raise RuntimeError(
                "RUNPOD_API_KEY is not set inside the pod. Without it nothing in this "
                "process can stop the pod, so every failsafe would be inert and the run "
                "would bill until a human noticed. Pass it through at provision time, "
                "or pass dry_run=True if you are deliberately checking wiring locally."
            )
        runpod = RunPodClient(runpod_key)

    wandb_client = WandbClient(os.environ.get("WANDB_API_KEY"))
    notifier = Notifier(cfg)
    diagnostics = Diagnostics(cfg, runpod=runpod, wandb=wandb_client)

    ladder = EscalationLadder(
        cfg,
        notifier=notifier,
        diagnostics=diagnostics,
        runpod=runpod,
        wandb_module=wandb_module,
        trainer=trainer,
        snapshot_dir=str(Path(cfg.expect.checkpoint_dir) / "snapshots"),
    )
    # SIGTERM is what RunPod sends on a stop, and what a `docker stop` sends.
    # Routing it through shutdown() is the difference between a graceful
    # checkpoint and losing the run's last hours.
    install_signal_handlers(ladder)

    baselines: dict[str, float] = {}
    if not skip_gate:
        sctx = StartupContext(
            ctx=Context(cfg=cfg, wandb=wandb_client, runpod=runpod),
            hardware=SystemHardware(),
            hooks=hooks or StartupHooks(),
            notifier=notifier,
        )

        def log_result(payload: dict) -> None:
            # The sentinel treats the absence of startup_ok within
            # startup.deadline_s as a failure. It is the only signal that can
            # catch a crash before wandb.init(), which W&B never sees.
            if wandb_module is not None and getattr(wandb_module, "run", None):
                wandb_module.log(payload)
            log.info("startup gate passed: %s", payload)

        result = gate_and_enforce(sctx, ladder=ladder, log_result=log_result)
        if not result.passed:
            raise StartupGateFailed(
                f"{result.summary()} -- {result.failure.detail if result.failure else ''}"
            )
        if result.unwired:
            log.warning("startup checks not wired (reported, not passed): %s",
                        ", ".join(result.unwired))
        baselines = result.baselines

    watchdog = Watchdog(
        cfg,
        ladder=ladder,
        wandb_client=wandb_client,
        runpod_client=runpod,
        gen_backend_health=gen_backend_health,
        baselines=baselines,
        interval_s=interval_s,
    )

    # Watchdog.__exit__ is the shutdown guard: completion, exception and
    # KeyboardInterrupt all terminate the pod, and the exception is re-raised.
    with watchdog:
        yield watchdog


__all__ = ["StartupGateFailed", "StartupHooks", "attach"]
