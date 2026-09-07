"""RLM regime probes (spec section 5) and the shared health/cost probes."""

from __future__ import annotations

import pytest

from rlmwatch.config import from_dict
from rlmwatch.probes.cost import CostPerProgressProbe, SpendProbe, WallClockProbe
from rlmwatch.probes.health import (
    GradNormProbe,
    LossFiniteProbe,
    ProjectedCompletionProbe,
    ThroughputProbe,
    trend,
)
from rlmwatch.probes.rlm import (
    RL_PROBES,
    SFT_PROBES,
    ClipFractionProbe,
    CompletionLengthProbe,
    EntropyProbe,
    FormatSuccessProbe,
    GenerationBackendProbe,
    HeldOutEvalProbe,
    KLDivergenceProbe,
    LossPlateauProbe,
    ReasoningTokenAccuracyProbe,
    ReplDegeneracyProbe,
    RewardHackingProbe,
    RewardModelLatencyProbe,
    RewardStdProbe,
    RolloutDurationProbe,
    SequenceLengthProbe,
    VramHeadroomProbe,
    probes_for_regime,
)
from tests.conftest import log_series
from tests.fakes.fake_wandb import gpu_system_metrics


def test_trend_slope():
    assert trend([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)
    assert trend([4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert trend([2.0]) == 0.0


class TestRewardStd:
    """The highest-value probe in the library."""

    def test_healthy_variance_is_ok(self, make_ctx, wandb_run):
        log_series(wandb_run, "reward/std", [0.4] * 25)
        assert RewardStdProbe().check(make_ctx()).status == "ok"

    def test_sustained_zero_variance_fails(self, make_ctx, wandb_run):
        """Advantage is zero, gradient is zero, every dashboard looks normal."""
        log_series(wandb_run, "reward/std", [0.0] * 20)
        verdict = RewardStdProbe().check(make_ctx())
        assert verdict.status == "fail"
        assert "no-op" in verdict.detail
        assert verdict.evidence["collapsed_of_window"] == 20

    def test_collapse_shorter_than_patience_only_warns(self, make_ctx, wandb_run):
        """One degenerate batch is normal; twenty in a row is not."""
        log_series(wandb_run, "reward/std", [0.4] * 15 + [0.0] * 3)
        assert RewardStdProbe().check(make_ctx()).status == "warn"

    def test_patience_is_configurable(self, make_ctx, config_dict, wandb_run):
        config_dict["health"] = {"reward_std_patience": 5}
        log_series(wandb_run, "reward/std", [0.0] * 5)
        assert RewardStdProbe().check(make_ctx(cfg=from_dict(config_dict))).status == "fail"

    def test_no_history_is_unknown(self, make_ctx):
        assert RewardStdProbe().check(make_ctx()).status == "unknown"

    def test_metric_alias_is_honoured(self, make_ctx, config_dict, wandb_run):
        """Trainers name this differently; the map lives in config, not a patch."""
        config_dict["health"] = {"metric_aliases": {"reward/std": "grpo/rew_std"},
                                 "reward_std_patience": 5}
        log_series(wandb_run, "grpo/rew_std", [0.0] * 5)
        assert RewardStdProbe().check(make_ctx(cfg=from_dict(config_dict))).status == "fail"


class TestKLDivergence:
    def test_within_limit_is_ok(self, make_ctx, wandb_run):
        log_series(wandb_run, "kl/ref", [0.05] * 20)
        assert KLDivergenceProbe().check(make_ctx()).status == "ok"

    def test_over_limit_fails(self, make_ctx, wandb_run):
        log_series(wandb_run, "kl/ref", [0.2] * 20)
        assert KLDivergenceProbe().check(make_ctx()).status == "fail"

    def test_steep_rise_warns_before_it_arrives(self, make_ctx, wandb_run):
        log_series(wandb_run, "kl/ref", [0.02 * i for i in range(1, 8)])
        assert KLDivergenceProbe().check(make_ctx()).status == "warn"


class TestEntropy:
    def test_healthy_entropy_is_ok(self, make_ctx, wandb_run):
        log_series(wandb_run, "policy/entropy", [0.9] * 20)
        assert EntropyProbe().check(make_ctx()).status == "ok"

    def test_below_floor_fails(self, make_ctx, wandb_run):
        log_series(wandb_run, "policy/entropy", [0.1] * 20)
        verdict = EntropyProbe().check(make_ctx())
        assert verdict.status == "fail"
        assert "exploration is dead" in verdict.detail

    def test_monotone_decline_warns_while_still_above_the_floor(self, make_ctx, wandb_run):
        log_series(wandb_run, "policy/entropy", [1.0 - 0.05 * i for i in range(12)])
        assert EntropyProbe().check(make_ctx()).status == "warn"


class TestFormatSuccess:
    def test_high_parse_rate_is_ok(self, make_ctx, wandb_run):
        log_series(wandb_run, "format/success_rate", [0.98] * 10)
        assert FormatSuccessProbe().check(make_ctx()).status == "ok"

    def test_scorer_returning_silent_zeros_fails(self, make_ctx, wandb_run):
        """A leftover llm_query reference kills every rollout on NameError."""
        log_series(wandb_run, "format/success_rate", [0.02] * 10)
        verdict = FormatSuccessProbe().check(make_ctx())
        assert verdict.status == "fail"
        assert "returning zeros" in verdict.detail


class TestRolloutDuration:
    def test_normal_rollout_is_ok(self, make_ctx, wandb_run):
        log_series(wandb_run, "rollout/duration_s", [210.0] * 10)
        ctx = make_ctx(baselines={"rollout": 200.0})
        assert RolloutDurationProbe().check(ctx).status == "ok"

    def test_ten_times_baseline_fails(self, make_ctx, wandb_run):
        log_series(wandb_run, "rollout/duration_s", [2000.0] * 10)
        ctx = make_ctx(baselines={"rollout": 200.0})
        verdict = RolloutDurationProbe().check(ctx)
        assert verdict.status == "fail"
        assert verdict.evidence["ratio"] == 10.0

    def test_no_baseline_is_unknown_not_a_guess(self, make_ctx, wandb_run):
        log_series(wandb_run, "rollout/duration_s", [2000.0])
        assert RolloutDurationProbe().check(make_ctx()).status == "unknown"


class TestGenerationBackend:
    def test_healthy_backend_is_ok(self, make_ctx):
        ctx = make_ctx(local={"gen_backend_healthy": True, "gen_backend_url": "x"})
        assert GenerationBackendProbe().check(ctx).status == "ok"

    def test_dead_backend_fails_rather_than_waiting_forever(self, make_ctx):
        ctx = make_ctx(local={"gen_backend_healthy": False})
        verdict = GenerationBackendProbe().check(ctx)
        assert verdict.status == "fail"
        assert "indefinitely" in verdict.detail

    def test_unreported_is_unknown(self, make_ctx):
        assert GenerationBackendProbe().check(make_ctx()).status == "unknown"


class TestShapeProbes:
    def test_completion_length_flags_both_ends(self, make_ctx, wandb_run):
        log_series(wandb_run, "completion/length", [9000.0])
        assert CompletionLengthProbe().check(make_ctx()).status == "warn"
        log_series(wandb_run, "completion/length", [10.0])
        assert CompletionLengthProbe().check(make_ctx()).status == "warn"

    def test_completion_length_in_band_is_ok(self, make_ctx, wandb_run):
        log_series(wandb_run, "completion/length", [1500.0])
        assert CompletionLengthProbe().check(make_ctx()).status == "ok"

    def test_clip_fraction_over_limit_warns(self, make_ctx, wandb_run):
        log_series(wandb_run, "clip_fraction", [0.5] * 20)
        assert ClipFractionProbe().check(make_ctx()).status == "warn"

    def test_vram_headroom_warns_as_it_trends_down(self, make_ctx, wandb_run):
        wandb_run.summary.update(gpu_system_metrics(90.0, mem_allocated_pct=97.0))
        verdict = VramHeadroomProbe().check(make_ctx())
        assert verdict.status == "warn"
        assert verdict.evidence["min_free_pct"] == pytest.approx(3.0)

    def test_reward_model_errors_fail(self, make_ctx, wandb_run):
        log_series(wandb_run, "reward_model/error_rate", [0.4] * 10)
        verdict = RewardModelLatencyProbe().check(make_ctx())
        assert verdict.status == "fail"
        assert "silently degrading" in verdict.detail


class TestRewardHacking:
    def test_reward_up_kl_up_eval_flat_is_the_signature(self, make_ctx, wandb_run):
        for i in range(20):
            wandb_run.log({"reward/mean": 0.2 + 0.01 * i, "kl/ref": 0.01 + 0.002 * i})
        log_series(wandb_run, "eval/score", [0.5, 0.49, 0.48])
        verdict = RewardHackingProbe().check(make_ctx())
        assert verdict.status == "warn"
        assert "reward-hacking signature" in verdict.detail

    def test_reward_and_eval_both_rising_is_fine(self, make_ctx, wandb_run):
        for i in range(20):
            wandb_run.log({"reward/mean": 0.2 + 0.01 * i, "kl/ref": 0.01 + 0.002 * i})
        log_series(wandb_run, "eval/score", [0.4, 0.5, 0.6])
        assert RewardHackingProbe().check(make_ctx()).status == "ok"


class TestReplDegeneracy:
    """The pre-registered kill criterion for this specific experiment."""

    def test_truncate_and_answer_strategy_fails(self, make_ctx, wandb_run):
        for i in range(20):
            wandb_run.log(
                {
                    "reward/mean": 0.2 + 0.01 * i,
                    "repl/mean_turns": 6.0 - 0.2 * i,
                    "repl/nontrivial_fraction": 0.8 - 0.03 * i,
                }
            )
        verdict = ReplDegeneracyProbe().check(make_ctx())
        assert verdict.status == "fail"
        assert "pre-registered kill criterion" in verdict.detail

    def test_reward_rising_with_repl_use_rising_is_the_result_we_want(
        self, make_ctx, wandb_run
    ):
        for i in range(20):
            wandb_run.log(
                {
                    "reward/mean": 0.2 + 0.01 * i,
                    "repl/mean_turns": 3.0 + 0.1 * i,
                    "repl/nontrivial_fraction": 0.5 + 0.01 * i,
                }
            )
        assert ReplDegeneracyProbe().check(make_ctx()).status == "ok"


class TestHealthProbes:
    def test_nan_loss_fails_immediately(self, make_ctx, wandb_run):
        log_series(wandb_run, "train/loss", [1.0, 0.9, float("nan")])
        verdict = LossFiniteProbe().check(make_ctx())
        assert verdict.status == "fail"
        assert "poisoned" in verdict.detail

    def test_finite_loss_is_ok(self, make_ctx, wandb_run):
        log_series(wandb_run, "train/loss", [1.0, 0.9, 0.8])
        assert LossFiniteProbe().check(make_ctx()).status == "ok"

    def test_grad_norm_collapse_warns(self, make_ctx, wandb_run):
        log_series(wandb_run, "train/grad_norm", [1e-12] * 5)
        assert GradNormProbe().check(make_ctx()).status == "warn"

    def test_grad_norm_in_band_is_ok(self, make_ctx, wandb_run):
        log_series(wandb_run, "train/grad_norm", [0.7] * 5)
        assert GradNormProbe().check(make_ctx()).status == "ok"

    def test_throughput_degradation_warns(self, make_ctx, wandb_run):
        log_series(wandb_run, "train/throughput", [400.0] * 10)
        ctx = make_ctx(baselines={"throughput": 1000.0})
        verdict = ThroughputProbe().check(ctx)
        assert verdict.status == "warn"
        assert verdict.evidence["drop_pct"] == 60.0

    def test_projected_completion_over_budget_warns(self, make_ctx):
        # 1h elapsed at step 10 of 250 projects to 25h, well past the 3h budget.
        ctx = make_ctx(local={"step": 10, "max_steps": 250})
        assert ProjectedCompletionProbe().check(ctx).status == "warn"


class TestCostProbes:
    def test_spend_under_cap_is_ok(self, make_ctx):
        assert SpendProbe().check(make_ctx()).status == "ok"

    def test_spend_reaching_the_cap_fails(self, make_ctx, runpod_server):
        runpod_server.pods["pod-1"].uptime_s = 6 * 3600  # 6h x $0.88 > $5 cap
        verdict = SpendProbe().check(make_ctx())
        assert verdict.status == "fail"
        assert verdict.evidence["spend_usd"] == pytest.approx(5.28)

    def test_spend_warns_at_the_configured_percentage(self, make_ctx, runpod_server):
        runpod_server.pods["pod-1"].uptime_s = 4.5 * 3600  # $3.96 = 79% of $5
        assert SpendProbe().check(make_ctx()).status == "warn"

    def test_wall_clock_cap_is_independent_of_spend(self, make_ctx, runpod_server):
        runpod_server.pods["pod-1"].uptime_s = 4 * 3600  # past the 3h cap
        assert WallClockProbe().check(make_ctx()).status == "fail"

    def test_every_cost_verdict_carries_the_projected_number(self, make_ctx):
        """An operator at 2am needs the figure in the message, not a link."""
        ctx = make_ctx(local={"step": 5, "max_steps": 20})
        verdict = CostPerProgressProbe().check(ctx)
        assert "projected_total_usd" in verdict.evidence
        assert "spend_usd" in verdict.evidence

    def test_cost_per_progress_flags_a_run_that_got_slower(self, make_ctx):
        ctx = make_ctx(local={"step": 2, "max_steps": 20},
                       baselines={"cost_per_pct": 0.01})
        verdict = CostPerProgressProbe().check(ctx)
        assert verdict.status == "warn"


class TestRegimeSelection:
    def test_rl_regime_loads_the_rl_probe_set(self):
        assert probes_for_regime("rl") is RL_PROBES

    def test_sft_regime_loads_the_sft_probe_set(self):
        assert probes_for_regime("sft") is SFT_PROBES

    def test_reward_std_is_rl_only(self):
        assert "rlm.reward_std" in {p.name for p in RL_PROBES}
        assert "rlm.reward_std" not in {p.name for p in SFT_PROBES}


class TestSftProbes:
    def test_loss_plateau_warns(self, make_ctx, config_dict, wandb_run):
        config_dict["health"] = {"plateau_window": 10}
        log_series(wandb_run, "train/loss", [0.5] * 10)
        assert LossPlateauProbe().check(make_ctx(cfg=from_dict(config_dict))).status == "warn"

    def test_formatting_not_reasoning_warns(self, make_ctx, wandb_run):
        for i in range(20):
            wandb_run.log({"train/loss": 1.0 - 0.02 * i,
                           "train/reasoning_token_accuracy": 0.5})
        verdict = ReasoningTokenAccuracyProbe().check(make_ctx())
        assert verdict.status == "warn"
        assert "formatting, not reasoning" in verdict.detail

    def test_sequence_length_near_max_warns_about_silent_truncation(self, make_ctx, wandb_run):
        log_series(wandb_run, "train/seq_len_p95", [3900.0])
        ctx = make_ctx(local={"max_seq_len": 4096})
        verdict = SequenceLengthProbe().check(ctx)
        assert verdict.status == "warn"
        assert "conclusions are" in verdict.detail

    def test_held_out_eval_decline_suggests_halting(self, make_ctx, wandb_run):
        log_series(wandb_run, "eval/score", [0.7, 0.65, 0.62, 0.60])
        verdict = HeldOutEvalProbe().check(make_ctx())
        assert verdict.status == "warn"
        assert "overfitting" in verdict.detail


def test_every_probe_returns_unknown_when_wandb_is_unreachable(make_ctx, wandb_api):
    """No probe may escalate to fail because the monitor went blind."""
    wandb_api.unavailable = True
    ctx = make_ctx(baselines={"rollout": 10.0, "throughput": 100.0},
                   local={"step": 1, "max_steps": 10, "max_seq_len": 4096})
    for probe in RL_PROBES + SFT_PROBES:
        if probe.name == "rlm.generation_backend":
            continue  # reads ctx.local only, not W&B
        assert probe.check(ctx).status == "unknown", probe.name
