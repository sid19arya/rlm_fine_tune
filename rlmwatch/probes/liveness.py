"""Liveness probes -- is anything happening at all? (spec section 4.1)

The premise: **run state alone is neither necessary nor sufficient.** W&B's
heartbeat is emitted by its own service process, so a deadlocked training loop
reports `Running` indefinitely; and a long network outage can mark a perfectly
healthy run `Crashed`. Last-progress-age is the real liveness signal, and the
rest of these probes exist to cross-check it from angles that fail differently.

The most valuable alert in the file is `ObserverAgreementProbe`: pod `RUNNING`
while W&B says `Crashed` means the process died but the container survived. That
is the expensive silent case -- nothing is training, nothing is complaining, and
the meter runs until someone notices.
"""

from __future__ import annotations

from rlmwatch.clients.wandb import TERMINAL_BAD_STATES, WandbUnavailable
from rlmwatch.probes.base import BaseProbe, Context, Verdict


class ProgressAgeProbe(BaseProbe):
    """Seconds since the training loop last logged anything.

    Compared against a *phase-aware* threshold: a GRPO rollout legitimately runs
    for many minutes with nothing logged, so the budget for `rollout` is derived
    from the warm-up baseline rather than shared with `update`.
    """

    name = "liveness.progress_age"

    def check(self, ctx: Context) -> Verdict:
        run_path = ctx.cfg.run.wandb
        if not run_path:
            return self.unknown("no W&B run configured", run=run_path)

        threshold = ctx.stall_threshold_s()
        try:
            age = ctx.wandb.last_progress_age(run_path)
        except WandbUnavailable as exc:
            # Blind, not broken. Sustained blindness is promoted to warn by the
            # escalation ladder; a single unreadable poll must not kill a pod.
            return self.unknown(f"cannot read progress: {exc}", phase=ctx.phase,
                                threshold_s=threshold)

        evidence = {"age_s": round(age, 1), "threshold_s": threshold, "phase": ctx.phase}
        if age > threshold:
            return self.fail(
                f"no progress logged for {age:.0f}s in phase {ctx.phase or 'unknown'} "
                f"(threshold {threshold:.0f}s)",
                **evidence,
            )
        if age > threshold * 0.6:
            return self.warn(f"progress ageing: {age:.0f}s of {threshold:.0f}s", **evidence)
        return self.ok(f"last progress {age:.0f}s ago", **evidence)


class RunStateProbe(BaseProbe):
    """W&B run state. A supporting signal, never the sole basis for a kill."""

    name = "liveness.run_state"

    def check(self, ctx: Context) -> Verdict:
        run_path = ctx.cfg.run.wandb
        if not run_path:
            return self.unknown("no W&B run configured")
        try:
            state = ctx.wandb.state(run_path)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read run state: {exc}")

        if state in TERMINAL_BAD_STATES:
            return self.fail(f"W&B run state is {state}", state=state)
        if state == "finished":
            # Not an error -- but the container does not stop when the script
            # does, and the meter keeps running until something terminates it.
            return self.warn(
                "run finished; the pod does NOT stop on its own and is still billing",
                state=state,
            )
        if state == "pending":
            return self.warn("run still pending", state=state)
        return self.ok(f"run state {state}", state=state)


class GpuUtilizationProbe(BaseProbe):
    """Mean GPU utilisation from W&B's own system metrics.

    Free cross-check: W&B collects this with no instrumentation. Sustained ~0%
    while the run claims to be training means the process is wedged or the work
    never started, and it costs nothing extra to notice.
    """

    name = "liveness.gpu_util"

    def check(self, ctx: Context) -> Verdict:
        run_path = ctx.cfg.run.wandb
        if not run_path:
            return self.unknown("no W&B run configured")
        try:
            state = ctx.wandb.state(run_path)
            metrics = ctx.wandb.system_metrics(run_path)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read system metrics: {exc}")

        util = metrics.mean_util()
        if util is None:
            return self.unknown("no GPU system metrics reported yet")
        floor = ctx.cfg.health.gpu_util_min_pct
        evidence = {"mean_util_pct": round(util, 1), "min_pct": floor,
                    "per_gpu": metrics.gpu_util_pct, "run_state": state}

        if state != "running":
            return self.ok(f"GPU util {util:.1f}% (run not running)", **evidence)
        if util < floor:
            return self.fail(
                f"GPUs idle at {util:.1f}% while the run reports Running -- paying for "
                f"nothing", **evidence,
            )
        return self.ok(f"GPU util {util:.1f}%", **evidence)


class PodStatusProbe(BaseProbe):
    """Is the pod itself up? Sentinel-only -- the watchdog dies with the pod."""

    name = "liveness.pod_status"

    def check(self, ctx: Context) -> Verdict:
        pod_id = ctx.cfg.run.pod_id
        if not pod_id:
            return self.unknown("no pod_id configured")
        try:
            status = ctx.runpod.status(pod_id)
        except Exception as exc:  # noqa: BLE001 - any client failure means "blind"
            return self.unknown(f"cannot read pod status: {exc}", pod_id=pod_id)

        evidence = {"pod_state": status.state, "gpu_count": status.gpu_count,
                    "uptime_s": status.uptime_s, "cost_per_hr": status.cost_per_hr}
        if not status.is_running:
            return self.fail(f"pod is {status.state}", **evidence)
        return self.ok(f"pod RUNNING for {status.uptime_s / 3600:.1f}h", **evidence)


class ObserverAgreementProbe(BaseProbe):
    """Cross-check the two observers against each other.

    **The single most valuable alert this library produces.** Pod `RUNNING` +
    W&B `Crashed` means the training process died but the container survived:
    nothing trains, nothing complains, and billing continues at full rate until
    a human happens to look. Neither observer can see this alone -- it exists
    only in the disagreement.
    """

    name = "liveness.observer_agreement"

    def check(self, ctx: Context) -> Verdict:
        pod_id, run_path = ctx.cfg.run.pod_id, ctx.cfg.run.wandb
        if not pod_id or not run_path:
            return self.unknown("need both pod_id and W&B run to compare observers")

        try:
            pod_running = ctx.runpod.status(pod_id).is_running
        except Exception as exc:  # noqa: BLE001
            return self.unknown(f"cannot read pod status: {exc}")
        try:
            state = ctx.wandb.state(run_path)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read run state: {exc}")

        evidence = {"pod_running": pod_running, "wandb_state": state}

        if pod_running and state in TERMINAL_BAD_STATES:
            return self.fail(
                f"pod is RUNNING but W&B says {state}: the training process died and the "
                f"container survived it. Nothing is training and the meter is still on.",
                **evidence,
            )
        if pod_running and state == "finished":
            return self.fail(
                "pod is RUNNING but the run has Finished: a clean exit does not stop "
                "billing, the container restarts. Terminate explicitly.",
                **evidence,
            )
        if not pod_running and state == "running":
            return self.warn(
                "W&B still reports Running but the pod is not RUNNING -- stale run "
                "state, the pod is already gone", **evidence,
            )
        return self.ok(f"observers agree (pod running={pod_running}, W&B {state})", **evidence)


class StepCounterProbe(BaseProbe):
    """In-process step counter. Watchdog-only.

    Independent of W&B entirely, so it still works when the network is the thing
    that is broken. The watchdog supplies `step` and `step_updated_at` in
    `ctx.local`.
    """

    name = "liveness.step_counter"

    def check(self, ctx: Context) -> Verdict:
        step = ctx.local.get("step")
        updated_at = ctx.local.get("step_updated_at")
        if step is None or updated_at is None:
            return self.unknown("training loop has not reported a step yet")

        age = ctx.local.get("now", 0.0) - float(updated_at)
        threshold = ctx.stall_threshold_s()
        evidence = {"step": step, "age_s": round(age, 1), "threshold_s": threshold,
                    "phase": ctx.phase}
        if age > threshold:
            return self.fail(
                f"step counter stuck at {step} for {age:.0f}s (threshold {threshold:.0f}s)",
                **evidence,
            )
        return self.ok(f"step {step}, advanced {age:.0f}s ago", **evidence)


#: Probes the external sentinel runs. Deliberately excludes StepCounterProbe,
#: which needs in-process state the sentinel cannot see.
SENTINEL_PROBES: tuple[BaseProbe, ...] = (
    PodStatusProbe(),
    RunStateProbe(),
    ObserverAgreementProbe(),
    ProgressAgeProbe(),
    GpuUtilizationProbe(),
)

#: Probes the in-pod watchdog runs. Excludes PodStatusProbe and
#: ObserverAgreementProbe: a process inside the pod cannot report the pod's own
#: death, and pretending otherwise would be false assurance.
WATCHDOG_PROBES: tuple[BaseProbe, ...] = (
    StepCounterProbe(),
    ProgressAgeProbe(),
)
