"""Tests for the V0 experiment scripts.

These scripts run once, on a live pod, with money already burning. That is the
worst possible place to discover a bug in them, so the logic that decides
"is this safe to launch" and "is this safe to terminate" is tested here.

The one under real scrutiny is `strip_sub_lm_calls`. Getting it wrong does not
crash the run -- it produces a clean-looking null result from a run that was
silently answering a different question.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "v0_smoke"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


strip = _load("strip_sub_lm_calls")
interventions = _load("interventions")
provision = _load("provision")


@pytest.fixture
def tree(tmp_path):
    """A miniature rlm_train tree with a globals binding and a prompt mention."""
    root = tmp_path / "rlm_train"
    root.mkdir()
    (root / "repl.py").write_text(
        "from rlm.sub import llm_query\n"
        "\n"
        "\n"
        "def build_globals(context):\n"
        "    env = {}\n"
        '    env["context"] = context\n'
        '    env["llm_query"] = llm_query\n'
        "    return env\n",
        encoding="utf-8",
    )
    (root / "prompts.py").write_text(
        'SYSTEM_PROMPT = """You have a Python REPL.\n'
        "`context` holds the document. You can also call llm_query(prompt) to ask\n"
        "a sub-LM for help with a chunk.\n"
        'Emit code in a ```repl block."""\n',
        encoding="utf-8",
    )
    return root


class TestStripSubLmCalls:
    def test_finds_both_the_globals_and_the_prompt(self, tree):
        report = strip.scan(tree)
        assert len(report.globals_) == 2  # the import and the env binding
        assert report.prompts, "the prompt mention must be found too"
        assert not report.clean

    def test_apply_edits_code_bindings(self, tree):
        strip.apply(strip.scan(tree))
        source = (tree / "repl.py").read_text(encoding="utf-8")
        assert "# from rlm.sub import llm_query" in source
        assert strip.MARKER in source

    def test_apply_never_touches_prompt_text(self, tree):
        """Commenting inside a triple-quoted string does not remove it from the
        prompt -- it puts a '#' in the string and the model still reads it."""
        before = (tree / "prompts.py").read_text(encoding="utf-8")
        strip.apply(strip.scan(tree))
        assert (tree / "prompts.py").read_text(encoding="utf-8") == before

    def test_a_commented_prompt_line_is_still_reported(self, tree):
        """The trap: a '#' inside a prompt block must not read as 'handled'."""
        (tree / "prompts.py").write_text(
            'SYSTEM_PROMPT = """You have a Python REPL.\n'
            "# You can also call llm_query(prompt) to ask a sub-LM.\n"
            'Emit code in a ```repl block."""\n',
            encoding="utf-8",
        )
        report = strip.scan(tree)
        assert report.prompts, "a '#' inside a prompt string is text, not a comment"

    def test_a_genuinely_commented_out_binding_is_not_reported(self, tree):
        (tree / "repl.py").write_text(
            "# from rlm.sub import llm_query  # removed\n"
            "def build_globals(context):\n"
            '    return {"context": context}\n',
            encoding="utf-8",
        )
        assert strip.scan(tree).globals_ == []

    def test_a_dict_key_binding_is_a_global_not_prompt_text(self, tree):
        """{"llm_query": fn} is the binding that matters most; it must not be
        misfiled as prompt text just because the key is a string."""
        (tree / "repl.py").write_text(
            "def build_globals(context):\n"
            '    return {"context": context, "llm_query": llm_query}\n',
            encoding="utf-8",
        )
        report = strip.scan(tree)
        assert [h.kind for h in report.globals_] == ["global"]

    def test_backups_are_written_before_editing(self, tree):
        strip.apply(strip.scan(tree))
        assert (tree / "repl.py.bak").exists()

    def test_backups_are_not_rescanned(self, tree):
        strip.apply(strip.scan(tree))
        report = strip.scan(tree)
        assert not any(".bak" in str(h.path) for h in report.globals_ + report.prompts)

    def test_asymmetric_state_is_named_explicitly(self, tree):
        """Globals stripped, prompt left: worse than doing neither."""
        strip.apply(strip.scan(tree))
        report = strip.scan(tree)
        assert report.globals_ == []
        assert report.prompts
        assert report.asymmetric is True

    def test_fully_clean_tree_is_clean_and_not_asymmetric(self, tmp_path):
        root = tmp_path / "rlm_train"
        root.mkdir()
        (root / "repl.py").write_text(
            "def build_globals(context):\n"
            '    return {"context": context, "re": __import__("re")}\n',
            encoding="utf-8",
        )
        (root / "prompts.py").write_text(
            'SYSTEM_PROMPT = """You have a Python REPL. `context` holds the\n'
            'document. Emit code in a ```repl block."""\n',
            encoding="utf-8",
        )
        report = strip.scan(root)
        assert report.clean and not report.asymmetric

    def test_non_python_prompt_files_are_scanned(self, tree):
        (tree / "system.txt").write_text(
            "You may also call llm_query to ask another model.\n", encoding="utf-8"
        )
        assert any(h.path.name == "system.txt" for h in strip.scan(tree).prompts)

    def test_a_missing_root_is_an_error_not_a_clean_report(self, tmp_path):
        """Scanning the wrong tree and reporting 'clean' is the worst outcome."""
        with pytest.raises(FileNotFoundError, match="worst outcome"):
            strip.scan(tmp_path / "nope")

    def test_verify_exit_code_is_nonzero_while_references_remain(self, tree):
        assert strip.main(["--root", str(tree), "--verify"]) == 1

    def test_verify_exit_code_is_zero_once_clean(self, tmp_path):
        root = tmp_path / "rlm_train"
        root.mkdir()
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        assert strip.main(["--root", str(root), "--verify"]) == 0


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

    def test_trainer_and_inference_are_on_separate_gpus_in_one_pod(self):
        assert self.cfg["inference_gpu_ids"] == [0]
        assert self.cfg["trainer_gpu_ids"] == [1]
