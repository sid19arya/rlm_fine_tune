"""Liveness probes (spec section 4.1)."""

from __future__ import annotations

from rlmwatch.probes.liveness import (
    GpuUtilizationProbe,
    ObserverAgreementProbe,
    PodStatusProbe,
    ProgressAgeProbe,
    RunStateProbe,
    StepCounterProbe,
)
from tests.conftest import NOW, POD_ID, RUN_PATH
from tests.fakes.fake_wandb import gpu_system_metrics


class TestProgressAge:
    def test_fresh_progress_is_ok(self, make_ctx, wandb_run):
        wandb_run.summary["_timestamp"] = NOW - 10
        assert ProgressAgeProbe().check(make_ctx()).status == "ok"

    def test_stale_progress_fails_against_the_default_threshold(self, make_ctx, wandb_run):
        wandb_run.summary["_timestamp"] = NOW - 400
        verdict = ProgressAgeProbe().check(make_ctx())
        assert verdict.status == "fail"
        assert verdict.evidence["threshold_s"] == 300

    def test_a_long_rollout_is_not_a_false_alarm(self, make_ctx, wandb_run):
        """The reason phase-aware staleness exists at all.

        400s of silence trips the 300s default but is unremarkable for a rollout
        whose measured warm-up baseline was 200s (threshold 4x200 = 800s).
        """
        wandb_run.summary["_timestamp"] = NOW - 400
        ctx = make_ctx(phase="rollout", baselines={"rollout": 200.0})
        verdict = ProgressAgeProbe().check(ctx)
        assert verdict.status == "ok"
        assert verdict.evidence["threshold_s"] == 800.0

    def test_a_rollout_nearing_its_budget_warns_before_it_fails(self, make_ctx, wandb_run):
        """The ageing band: visible early, not yet actionable."""
        wandb_run.summary["_timestamp"] = NOW - 700
        ctx = make_ctx(phase="rollout", baselines={"rollout": 200.0})
        assert ProgressAgeProbe().check(ctx).status == "warn"

    def test_a_genuinely_hung_rollout_still_fails(self, make_ctx, wandb_run):
        wandb_run.summary["_timestamp"] = NOW - 5000
        ctx = make_ctx(phase="rollout", baselines={"rollout": 200.0})
        assert ProgressAgeProbe().check(ctx).status == "fail"

    def test_network_outage_is_unknown_not_fail(self, make_ctx, wandb_api):
        """A blip must never be able to terminate a healthy pod."""
        wandb_api.unavailable = True
        assert ProgressAgeProbe().check(make_ctx()).status == "unknown"


class TestRunState:
    def test_running_is_ok(self, make_ctx):
        assert RunStateProbe().check(make_ctx()).status == "ok"

    def test_crashed_fails(self, make_ctx, wandb_run):
        wandb_run.state = "crashed"
        assert RunStateProbe().check(make_ctx()).status == "fail"

    def test_finished_warns_because_the_pod_keeps_billing(self, make_ctx, wandb_run):
        wandb_run.state = "finished"
        verdict = RunStateProbe().check(make_ctx())
        assert verdict.status == "warn"
        assert "billing" in verdict.detail


class TestGpuUtilization:
    def test_busy_gpus_are_ok(self, make_ctx, wandb_run):
        wandb_run.summary.update(gpu_system_metrics(94.0))
        assert GpuUtilizationProbe().check(make_ctx()).status == "ok"

    def test_idle_gpus_while_running_fail(self, make_ctx, wandb_run):
        wandb_run.summary.update(gpu_system_metrics(1.0))
        verdict = GpuUtilizationProbe().check(make_ctx())
        assert verdict.status == "fail"
        assert verdict.evidence["mean_util_pct"] == 1.0

    def test_idle_gpus_on_a_finished_run_are_not_an_alert(self, make_ctx, wandb_run):
        wandb_run.state = "finished"
        wandb_run.summary.update(gpu_system_metrics(0.0))
        assert GpuUtilizationProbe().check(make_ctx()).status == "ok"

    def test_no_metrics_yet_is_unknown(self, make_ctx):
        assert GpuUtilizationProbe().check(make_ctx()).status == "unknown"


class TestPodStatus:
    def test_running_pod_is_ok(self, make_ctx):
        assert PodStatusProbe().check(make_ctx()).status == "ok"

    def test_exited_pod_fails(self, make_ctx, runpod_server):
        runpod_server.pods[POD_ID].state = "EXITED"
        assert PodStatusProbe().check(make_ctx()).status == "fail"

    def test_unreachable_api_is_unknown(self, make_ctx, runpod_server):
        runpod_server.transient_failures = 99
        assert PodStatusProbe().check(make_ctx()).status == "unknown"


class TestObserverAgreement:
    """The single most valuable alert in the library."""

    def test_agreement_is_ok(self, make_ctx):
        assert ObserverAgreementProbe().check(make_ctx()).status == "ok"

    def test_pod_running_while_wandb_crashed_is_the_expensive_silent_case(
        self, make_ctx, wandb_run
    ):
        wandb_run.state = "crashed"
        verdict = ObserverAgreementProbe().check(make_ctx())
        assert verdict.status == "fail"
        assert verdict.evidence == {"pod_running": True, "wandb_state": "crashed"}
        assert "container survived" in verdict.detail

    def test_pod_running_after_a_clean_finish_still_fails(self, make_ctx, wandb_run):
        """A clean exit does not stop billing; the container restarts."""
        wandb_run.state = "finished"
        verdict = ObserverAgreementProbe().check(make_ctx())
        assert verdict.status == "fail"
        assert "does not stop billing" in verdict.detail

    def test_dead_pod_with_stale_running_state_only_warns(self, make_ctx, runpod_server):
        runpod_server.pods[POD_ID].state = "EXITED"
        assert ObserverAgreementProbe().check(make_ctx()).status == "warn"

    def test_either_observer_blind_is_unknown(self, make_ctx, wandb_api):
        wandb_api.unavailable = True
        assert ObserverAgreementProbe().check(make_ctx()).status == "unknown"


class TestStepCounter:
    def test_advancing_steps_are_ok(self, make_ctx):
        ctx = make_ctx(local={"step": 12, "step_updated_at": 1000.0, "now": 1030.0})
        assert StepCounterProbe().check(ctx).status == "ok"

    def test_stuck_counter_fails_independently_of_wandb(self, make_ctx, wandb_api):
        """Works even when the network is the thing that is broken."""
        wandb_api.unavailable = True
        ctx = make_ctx(local={"step": 12, "step_updated_at": 1000.0, "now": 1400.0})
        verdict = StepCounterProbe().check(ctx)
        assert verdict.status == "fail"
        assert verdict.evidence["step"] == 12

    def test_no_step_reported_yet_is_unknown(self, make_ctx):
        assert StepCounterProbe().check(make_ctx()).status == "unknown"


def test_watchdog_probe_set_excludes_pod_level_checks():
    """A process inside the pod cannot report the pod's own death."""
    from rlmwatch.probes.liveness import SENTINEL_PROBES, WATCHDOG_PROBES

    watchdog_names = {p.name for p in WATCHDOG_PROBES}
    assert "liveness.pod_status" not in watchdog_names
    assert "liveness.observer_agreement" not in watchdog_names
    assert "liveness.pod_status" in {p.name for p in SENTINEL_PROBES}


def test_probes_report_their_own_identity(make_ctx):
    for probe in (ProgressAgeProbe(), RunStateProbe(), PodStatusProbe()):
        assert probe.check(make_ctx()).probe == probe.name


def test_no_wandb_run_configured_is_unknown(make_ctx, config_dict):
    from rlmwatch.config import from_dict

    config_dict["run"] = {**config_dict["run"], "wandb": ""}
    ctx = make_ctx(cfg=from_dict(config_dict))
    assert ProgressAgeProbe().check(ctx).status == "unknown"
    assert RUN_PATH not in ctx.cfg.run.wandb
