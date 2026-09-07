"""Config loading and validation.

The bar: a misconfiguration must be caught at load, not six hours in.
"""

from __future__ import annotations

import pytest

from rlmwatch.config import ConfigError, StallThreshold, from_dict, load_config

MINIMAL = {
    "run": {"name": "rlm-ft-001"},
    "budget": {"max_usd": 5.0, "hourly_rate_usd": 0.88},
    "failsafe": {"on_terminal": "stop"},
}


def test_minimal_config_loads_with_spec_defaults():
    cfg = from_dict(MINIMAL)
    assert cfg.run.regime == "rl"
    assert cfg.health.reward_std_min == 0.01
    assert cfg.health.reward_std_patience == 20
    assert cfg.failsafe.confirm_delay_s == 120.0


def test_unknown_key_is_rejected_because_a_typo_is_a_silent_failure():
    with pytest.raises(ConfigError, match="unknown key"):
        from_dict({**MINIMAL, "health": {"reward_std_minimum": 0.5}})


def test_unknown_section_is_rejected():
    with pytest.raises(ConfigError, match="unknown top-level section"):
        from_dict({**MINIMAL, "helth": {}})


def test_terminate_without_network_volume_is_refused():
    """The combination that silently throws away every checkpoint."""
    with pytest.raises(ConfigError, match="network volume"):
        from_dict({**MINIMAL, "failsafe": {"on_terminal": "terminate"}})


def test_terminate_is_allowed_once_the_volume_is_acknowledged():
    cfg = from_dict(
        {
            **MINIMAL,
            "failsafe": {
                "on_terminal": "terminate",
                "checkpoint_dir_is_network_volume": True,
            },
        }
    )
    assert cfg.failsafe.on_terminal == "terminate"


def test_dead_mans_timeout_must_outlast_confirmation():
    """Otherwise the backstop fires before a fail verdict can be confirmed."""
    with pytest.raises(ConfigError, match="dead_mans_timeout_s"):
        from_dict(
            {**MINIMAL, "failsafe": {"on_terminal": "stop", "confirm_delay_s": 600,
                                     "dead_mans_timeout_s": 300}}
        )


def test_budget_has_no_unlimited_setting():
    with pytest.raises(ConfigError, match="max_usd"):
        from_dict({**MINIMAL, "budget": {"max_usd": 0}})


def test_env_interpolation_and_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-abc123")
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "run:\n"
        "  name: rlm-ft-001\n"
        "  pod_id: ${RUNPOD_POD_ID}\n"
        "  wandb: ${WANDB_PATH:-me/proj/run1}\n"
        "budget:\n"
        "  max_usd: 5\n"
        "  hourly_rate_usd: 0.88\n"
        "failsafe:\n"
        "  on_terminal: stop\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.run.pod_id == "pod-abc123"
    assert cfg.run.wandb == "me/proj/run1"
    assert cfg.run.wandb_url == "https://wandb.ai/me/proj/runs/run1"


def test_missing_env_var_without_default_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE_NOT_SET", raising=False)
    path = tmp_path / "cfg.yaml"
    path.write_text("run:\n  name: x\n  pod_id: ${NOPE_NOT_SET}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="NOPE_NOT_SET"):
        load_config(path)


def test_wandb_path_shape_is_validated():
    with pytest.raises(ConfigError, match="entity/project/run-id"):
        from_dict({**MINIMAL, "run": {"name": "x", "wandb": "just-a-run"}})


class TestStallThreshold:
    """Phase-aware staleness is not optional for RL."""

    def test_unknown_phase_falls_back_to_default(self):
        st = StallThreshold(default=300)
        assert st.resolve("nonesuch", {}) == 300
        assert st.resolve(None, {}) == 300

    def test_explicit_phase_value_wins(self):
        st = StallThreshold(phases={"checkpoint": 900})
        assert st.resolve("checkpoint", {}) == 900

    def test_auto_scales_off_the_measured_baseline(self):
        st = StallThreshold(phases={"rollout": "auto"}, auto_multiplier=4, auto_floor_s=600)
        assert st.resolve("rollout", {"rollout": 400.0}) == 1600.0

    def test_auto_never_drops_below_the_floor(self):
        """A fast warm-up rollout must not produce a hair-trigger threshold."""
        st = StallThreshold(phases={"rollout": "auto"}, auto_multiplier=4, auto_floor_s=600)
        assert st.resolve("rollout", {"rollout": 10.0}) == 600.0

    def test_auto_without_a_baseline_uses_the_floor_not_the_default(self):
        st = StallThreshold(default=300, phases={"rollout": "auto"}, auto_floor_s=600)
        assert st.resolve("rollout", {}) == 600.0

    def test_non_numeric_phase_value_is_rejected(self):
        st = StallThreshold(phases={"rollout": "soon"})
        with pytest.raises(ConfigError, match="positive number or 'auto'"):
            st.validate()


def test_budget_spend_includes_storage():
    cfg = from_dict({**MINIMAL, "budget": {"max_usd": 100, "hourly_rate_usd": 1.0,
                                           "storage_gb": 100}})
    # 10h compute at $1/h, plus 100GB at $0.07/GB/month prorated over 10/730 h.
    assert cfg.budget.spend_at(10.0) == pytest.approx(10.0 + 7.0 * 10 / 730, rel=1e-6)


def test_metric_aliases_map_trainer_keys_without_patching_the_trainer():
    cfg = from_dict({**MINIMAL, "health": {"metric_aliases": {"reward/std": "grpo/rew_std"}}})
    assert cfg.health.key_for("reward/std") == "grpo/rew_std"
    assert cfg.health.key_for("kl/ref") == "kl/ref"
