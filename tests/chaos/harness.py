"""Fault-injection harness.

Composes the two fakes into a simulated run with a controllable clock, so the
section 8 faults can be injected deterministically and in seconds rather than
hours. Nothing here touches a real GPU, pod, or network.

The simulation is deliberately literal about the failure modes it reproduces:

* `kill -9` leaves the pod RUNNING and the W&B run stuck mid-step. W&B's own
  service process stops heartbeating shortly after, so the run eventually reads
  `crashed` -- but the pod does not notice at all.
* `SIGSTOP` leaves *everything* looking healthy. The pod is RUNNING, the W&B run
  says `running`, and wandb's service keeps heartbeating from a process that is
  not the training loop. Only progress age moves. This is the case that
  motivates the whole library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlmwatch.actions import EscalationLadder
from rlmwatch.clients.runpod import RunPodClient
from rlmwatch.clients.wandb import WandbClient
from rlmwatch.config import RunConfig, from_dict
from rlmwatch.diagnostics import Diagnostics
from rlmwatch.notify import Notifier, RecordingSink
from rlmwatch.sentinel import Sentinel
from rlmwatch.watchdog import Watchdog
from tests.fakes.fake_runpod import FakePod, FakeRunPodServer
from tests.fakes.fake_wandb import FakeRun, FakeWandbApi, gpu_system_metrics

START = 1_700_000_000.0
RUN_PATH = "acme/rlm/chaos-001"
POD_ID = "pod-chaos"

CHAOS_CONFIG: dict[str, Any] = {
    "run": {"name": "chaos", "regime": "rl", "wandb": RUN_PATH, "pod_id": POD_ID},
    "expect": {"gpu_type": "NVIDIA A40", "gpu_count": 2, "min_free_vram_gb": 40,
               "min_disk_gb": 50},
    "startup": {"deadline_s": 900},
    "budget": {"hourly_rate_usd": 0.88, "max_usd": 50.0, "max_wall_clock_h": 24.0},
    "failsafe": {
        "on_terminal": "stop",
        "confirm_delay_s": 120,
        "unknown_tolerance_s": 900,
        "dead_mans_timeout_s": 1800,
        "checkpoint_timeout_s": 600,
    },
    "stall_threshold": {"default": 300, "rollout": "auto", "checkpoint": 900},
    "poll_interval_s": 60,
}


class Clock:
    """Advanceable wall clock. Every component shares one."""

    def __init__(self, t: float = START) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t

    def sleep(self, seconds: float) -> None:
        """Stand-in for time.sleep -- advances the clock instead of blocking."""
        self.advance(seconds)


class FakeTrainer:
    """A cooperative trainer that can be made uncooperative."""

    def __init__(self, *, can_checkpoint: bool = True) -> None:
        self.can_checkpoint = can_checkpoint
        self.checkpoint_requests = 0
        self.checkpoints_saved = 0

    def request_checkpoint(self) -> None:
        self.checkpoint_requests += 1

    def wait_for_checkpoint(self, timeout_s: float) -> bool:
        if self.can_checkpoint:
            self.checkpoints_saved += 1
            return True
        return False


@dataclass
class ChaosRun:
    """One simulated run: pod, W&B run, clock, ladder, and both observers."""

    cfg: RunConfig
    clock: Clock
    pod: FakePod
    run: FakeRun
    runpod_server: FakeRunPodServer
    wandb_api: FakeWandbApi
    runpod: RunPodClient
    wandb: WandbClient
    sink: RecordingSink
    ladder: EscalationLadder
    trainer: FakeTrainer
    diagnostics: Diagnostics
    step: int = 0
    commands_run: list[list[str]] = field(default_factory=list)

    # --- driving a healthy run ------------------------------------------------

    def log_step(self, **metrics: float) -> None:
        """Advance one healthy training step, logging plausible RL metrics."""
        self.step += 1
        payload = {
            "train/loss": 1.0 - 0.001 * self.step,
            "train/grad_norm": 0.7,
            "reward/mean": 0.24 + 0.001 * self.step,
            "reward/std": 0.42,
            "kl/ref": 0.02,
            "policy/entropy": 0.9,
            "completion/length": 1200.0,
            "format/success_rate": 0.97,
            "rollout/duration_s": 200.0,
            "repl/mean_turns": 4.0,
            "repl/nontrivial_fraction": 0.7,
            "train/throughput": 1000.0,
        }
        payload.update(metrics)
        payload.update(gpu_system_metrics(93.0))
        self.run.log(payload, timestamp=self.clock())
        self.pod.uptime_s = self.clock() - START

    # --- fault injection ------------------------------------------------------

    def kill_9_trainer(self) -> None:
        """SIGKILL the trainer. The pod survives; nothing else notices.

        No handler runs -- no checkpoint, no wandb.finish, no terminate. The
        container stays up and keeps billing. W&B eventually marks the run
        crashed when its service process stops heartbeating.
        """
        self.run.state = "crashed"
        # Progress stops here; the pod stays RUNNING.

    def sigstop_trainer(self) -> None:
        """SIGSTOP the trainer. Everything continues to look healthy.

        Pod RUNNING, W&B run `running`, wandb's service still heartbeating from
        a process that is not the training loop. Only progress age moves.
        """
        # Nothing to change: the fault *is* the absence of new log rows.

    def cut_wandb_network(self) -> None:
        self.wandb_api.unavailable = True
        self.wandb.invalidate()

    def restore_wandb_network(self) -> None:
        self.wandb_api.unavailable = False
        self.wandb.invalidate()

    def cut_runpod_api(self) -> None:
        self.runpod_server.transient_failures = 10**6

    def restore_runpod_api(self) -> None:
        self.runpod_server.transient_failures = 0

    def finish_normally(self) -> None:
        self.run.state = "finished"

    # --- observers ------------------------------------------------------------

    def sentinel(self, **kwargs: Any) -> Sentinel:
        kwargs.setdefault("pod_created_at", START)
        return Sentinel(self.cfg, ladder=self.ladder, wandb_client=self.wandb,
                        runpod_client=self.runpod, now=self.clock,
                        sleep=self.clock.sleep, **kwargs)

    def watchdog(self, **kwargs: Any) -> Watchdog:
        return Watchdog(self.cfg, ladder=self.ladder, wandb_client=self.wandb,
                        runpod_client=self.runpod, now=self.clock, **kwargs)

    # --- assertions -----------------------------------------------------------

    @property
    def terminated(self) -> bool:
        return bool(self.runpod_server.stopped or self.runpod_server.terminated)

    @property
    def alerts(self) -> list:
        return self.sink.alerts

    def alert_probes(self) -> list[str]:
        return [a.verdict.probe for a in self.sink.alerts]

    def snapshots(self):
        return [r.snapshot for r in self.ladder.history if r.snapshot is not None]


def build_run(config_overrides: dict[str, Any] | None = None, *,
              startup_ok: bool = True) -> ChaosRun:
    """Assemble a healthy simulated run, ready to have faults injected."""
    raw = {k: (dict(v) if isinstance(v, dict) else v) for k, v in CHAOS_CONFIG.items()}
    for section, values in (config_overrides or {}).items():
        if isinstance(values, dict):
            raw[section] = {**raw.get(section, {}), **values}
        else:
            raw[section] = values
    cfg = from_dict(raw)

    clock = Clock()
    server = FakeRunPodServer()
    pod = server.add(FakePod(POD_ID, uptime_s=0.0, cost_per_hr=0.88))
    api = FakeWandbApi()
    summary: dict[str, Any] = {"_timestamp": START}
    if startup_ok:
        summary["startup_ok"] = 1
    run = api.add(FakeRun(RUN_PATH, state="running", summary=summary))

    runpod = RunPodClient("key", client=server.client(), sleep=lambda _: None)
    wandb_client = WandbClient(api=api, now=clock, cache_ttl_s=0.0)
    sink = RecordingSink()
    trainer = FakeTrainer()

    commands: list[list[str]] = []

    def runner(cmd: list[str], timeout: float = 20.0) -> str:
        commands.append(cmd)
        if cmd[0] == "py-spy":
            return (
                "Thread 0x7f (active)\n"
                "  _wait_for_rollout (rlm_train/orchestrator.py:214)\n"
                "  main (train.py:88)\n"
            )
        return f"<{cmd[0]} output>"

    diagnostics = Diagnostics(cfg, runpod=runpod, wandb=wandb_client, runner=runner,
                              pid_finder=lambda: [4242])
    ladder = EscalationLadder(
        cfg,
        notifier=Notifier(cfg, sinks=[sink]),
        diagnostics=diagnostics,
        runpod=runpod,
        trainer=trainer,
        sleep=clock.sleep,
        now=clock,
    )

    return ChaosRun(cfg=cfg, clock=clock, pod=pod, run=run, runpod_server=server,
                    wandb_api=api, runpod=runpod, wandb=wandb_client, sink=sink,
                    ladder=ladder, trainer=trainer, diagnostics=diagnostics,
                    commands_run=commands)
