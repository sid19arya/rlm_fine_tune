"""Tests for the V0 experiment scripts.

These scripts run once, on a live pod, with money already burning. That is the
worst possible place to discover a bug in them, so the logic that decides
"is this safe to launch" and "is this safe to terminate" is tested here.

The experimental condition itself is no longer enforced from this repo: it is
`enable_sub_lm` in the pinned rlm fork, which drives the REPL globals, the
prompt and the rubric gate from one flag. What is tested here is that
`verify_ablation.py` correctly refuses a checkout where that flag is absent or
not wired -- which is the case that would otherwise spend $5 measuring the
wrong experiment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "v0_smoke"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verify_ablation = _load("verify_ablation")
interventions = _load("interventions")
provision = _load("provision")


class TestProvision:
    def test_it_refuses_without_the_spend_cap_assertion(self, capsys):
        """Nothing can verify an account spending limit through the API."""
        assert provision.main([]) == 2
        assert "spending limit" in capsys.readouterr().err

    def test_the_spec_never_requests_a_network_volume(self):
        """It pins the datacenter and bills monthly, indefinitely."""
        assert "networkVolumeId" not in provision.POD_SPEC

    def test_the_spec_matches_the_v0_hardware(self):
        assert provision.POD_SPEC["gpuCount"] == 2
        assert provision.POD_SPEC["containerDiskInGb"] == 30
        assert provision.POD_SPEC["volumeInGb"] == 100
        assert provision.POD_SPEC["volumeMountPath"] == "/workspace"

    def test_a6000_is_the_documented_fallback_for_a40(self):
        names = [name for name, _ in provision.GPU_PREFERENCES]
        assert names[0] == "NVIDIA A40"
        assert "A6000" in names[1]

    def test_dry_run_redacts_secrets(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_API_KEY", "super-secret-key")
        monkeypatch.setenv("HF_TOKEN", "hf_secret")
        assert provision.main(["--spend-cap-confirmed", "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "super-secret-key" not in out and "hf_secret" not in out

    def test_missing_hf_token_is_caught_before_provisioning(self, monkeypatch, capsys):
        """Not at the end of a successful run, when the disk is about to die."""
        monkeypatch.setenv("RUNPOD_API_KEY", "k")
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert provision.main(["--spend-cap-confirmed"]) == 2
        assert "before the pod's volume disk is destroyed" in capsys.readouterr().err


class TestInterventionLog:
    def test_entries_round_trip(self, tmp_path):
        path = tmp_path / "interventions.jsonl"
        assert interventions.main([
            "--path", str(path), "--task", "3", "--category", "DEPS",
            "--tried", "uv pip install failed", "--human-did", "prebuilt wheel",
        ]) == 0
        rows = interventions.read(path)
        assert len(rows) == 1
        assert rows[0]["category"] == "DEPS"
        assert rows[0]["recoverable"] is True
        assert "ts" in rows[0]

    def test_entries_append_rather_than_overwrite(self, tmp_path):
        path = tmp_path / "interventions.jsonl"
        for task in (1, 2, 3):
            interventions.main(["--path", str(path), "--task", str(task),
                                "--category", "AUTH", "--tried", "x", "--human-did", "y"])
        assert len(interventions.read(path)) == 3

    def test_unrecoverable_entries_are_surfaced_in_the_summary(self, tmp_path):
        path = tmp_path / "interventions.jsonl"
        interventions.main(["--path", str(path), "--task", "2", "--category", "PROVISION",
                            "--tried", "requested A40", "--human-did", "used the console",
                            "--unrecoverable"])
        summary = interventions.summarise(interventions.read(path))
        assert "could not have recovered" in summary

    def test_an_empty_log_says_so_rather_than_implying_success(self, tmp_path):
        summary = interventions.summarise(interventions.read(tmp_path / "none.jsonl"))
        assert "no interventions logged" in summary
        assert "check that entries are being written" in summary

    def test_all_spec_categories_are_available(self):
        assert set(interventions.CATEGORIES) == {
            "AUTH", "PROVISION", "ORDERING", "DEPS", "CONFIG", "LONGRUN", "DIAGNOSE"
        }


class TestSmokeToml:
    """The three settings that exist to prevent the three known failures."""

    def setup_method(self):
        try:
            import tomllib
        except ImportError:  # pragma: no cover - py<3.11
            import tomli as tomllib
        self.cfg = tomllib.loads((EXPERIMENT_DIR / "smoke.toml").read_text(encoding="utf-8"))

    def test_attention_is_fa2_not_fa3(self):
        """flash_attention_3 is Hopper-only and will not build on the A40."""
        assert self.cfg["trainer"]["model"]["attn"] == "flash_attention_2"

    def test_hybrid_thinking_is_disabled(self):
        """Left on, trajectories spend their budget reasoning, never reaching the REPL."""
        extra = self.cfg["orchestrator"]["train"]["sampling"]["extra_body"]
        assert extra["enable_thinking"] is False

    def test_the_run_shape_matches_v0(self):
        assert self.cfg["max_steps"] == 20
        assert self.cfg["model"]["name"] == "Qwen/Qwen3-8B"
        orchestrator = self.cfg["orchestrator"]
        assert orchestrator["batch_size"] * orchestrator["rollouts_per_example"] == 32

    def test_trainer_and_inference_split_two_gpus_in_one_pod(self):
        """prime-rl needs the trainer and inference server on the same node."""
        deployment = self.cfg["deployment"]
        assert deployment["gpus_per_node"] == 2
        assert deployment["num_train_gpus"] == 1
        assert deployment["num_infer_gpus"] == 1

    def test_it_matches_the_prime_rl_schema_not_the_spec_prose(self):
        """Env kwargs only reach load_environment through this one table."""
        args = self.cfg["orchestrator"]["train"]["env"][0]["args"]
        assert args["enable_sub_lm"] is False
        assert args["dataset_name"] == "spam"
        assert args["min_subcall"] == 0
        assert self.cfg["orchestrator"]["train"]["env"][0]["id"] == "oolong"


class TestVerifyAblation:
    """Guards the case that would spend $5 measuring the wrong experiment."""

    def test_it_reads_the_arm_from_smoke_toml(self):
        flag = verify_ablation.read_flag(EXPERIMENT_DIR / "smoke.toml")
        assert flag is False, "smoke.toml must select the no-recursion arm"

    def test_a_config_without_the_flag_returns_none(self, tmp_path):
        config = tmp_path / "c.toml"
        config.write_text("max_steps = 20\n", encoding="utf-8")
        assert verify_ablation.read_flag(config) is None

    def test_it_refuses_a_checkout_without_the_flag(self, tmp_path):
        """Upstream rlm has no enable_sub_lm; running against it silently
        measures the with-recursion arm."""
        results = verify_ablation.check(False, tmp_path / "not-a-checkout")
        assert results and not all(ok for ok, _ in results)
        assert "cannot import rlm" in results[0][1]

    def test_all_four_sub_lm_names_are_checked(self):
        """The strip script only knew about llm_query; rlm binds four."""
        assert set(verify_ablation.SUB_LM_NAMES) == {
            "llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched"
        }

    def test_it_also_looks_for_prose_tells_not_just_function_names(self):
        """A prompt clean of the names but still describing sub-LLMs is the
        same failure in slower motion."""
        assert "sub-LLM" in verify_ablation.PROMPT_TELLS

    def test_the_flag_is_read_from_the_table_prime_rl_actually_uses(self, tmp_path):
        """Only [orchestrator.train.env.args] becomes load_environment kwargs."""
        config = tmp_path / "c.toml"
        config.write_text(
            "[orchestrator.train.env.args]\nenable_sub_lm = false\n", encoding="utf-8"
        )
        assert verify_ablation.read_flag(config) is False

    def test_a_flag_in_the_wrong_table_is_caught_as_dead_config(self, tmp_path):
        """It looks right to a reader and does nothing at all."""
        config = tmp_path / "c.toml"
        config.write_text("[env]\nenable_sub_lm = false\n", encoding="utf-8")
        assert verify_ablation.read_flag(config) is None
        assert verify_ablation.misplaced_flag(config) == "env"

    def test_the_real_smoke_toml_has_no_misplaced_flag(self):
        assert verify_ablation.misplaced_flag(EXPERIMENT_DIR / "smoke.toml") is None
