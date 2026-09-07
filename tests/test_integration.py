"""The wiring layer (spec section 9).

The constraint being tested is as much about what `attach()` does *not* need
from the training script as what it does: gate, watchdog, shutdown wrapper, and
nothing else.
"""

from __future__ import annotations

import pytest

import rlmwatch
from rlmwatch.config import from_dict
from rlmwatch.integration import StartupGateFailed, attach
from rlmwatch.probes.startup import StartupHooks
from tests.conftest import BASE_CONFIG


@pytest.fixture
def local_cfg(tmp_path):
    """A config whose checkpoint dir exists, so snapshots have somewhere to go."""
    raw = {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASE_CONFIG.items()}
    raw["expect"] = {**raw["expect"], "checkpoint_dir": str(tmp_path / "ckpt")}
    return from_dict(raw)


def test_attach_is_exported_at_the_top_level():
    """Three lines in train.py means one import."""
    assert rlmwatch.attach is attach


class TestDryRun:
    def test_dry_run_needs_no_runpod_credential(self, local_cfg):
        """Wiring can be checked on a laptop, with nothing able to kill a pod."""
        with attach(local_cfg, skip_gate=True, dry_run=True) as watch:
            assert watch.ladder.runpod is None

    def test_dry_run_cannot_terminate_anything(self, local_cfg):
        with attach(local_cfg, skip_gate=True, dry_run=True) as watch:
            ladder = watch.ladder
        assert ladder.terminated is False

    def test_a_missing_runpod_key_is_refused_outside_dry_run(self, local_cfg, monkeypatch):
        """Without it, every failsafe is inert and the pod bills unnoticed."""
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="RUNPOD_API_KEY"):
            with attach(local_cfg, skip_gate=True):
                pass


class TestTrainerContract:
    def test_the_loop_reports_phase_step_and_heartbeat(self, local_cfg):
        with attach(local_cfg, skip_gate=True, dry_run=True) as watch:
            watch.phase("rollout")
            watch.heartbeat()
            watch.step(1, max_steps=20)

            assert watch.context.phase == "rollout"
            assert watch.context.local["step"] == 1
            assert watch.context.local["max_steps"] == 20

    def test_halted_is_visible_to_the_training_loop(self, local_cfg):
        with attach(local_cfg, skip_gate=True, dry_run=True) as watch:
            assert watch.halted is False
            watch.ladder.halt_requested = True
            assert watch.halted is True

    def test_the_rl_probe_set_is_attached(self, local_cfg):
        with attach(local_cfg, skip_gate=True, dry_run=True) as watch:
            names = {p.name for p in watch.probes}
            assert "rlm.reward_std" in names
            assert "liveness.pod_status" not in names, \
                "the watchdog cannot report the pod's own death"


class TestShutdownGuard:
    def test_normal_completion_runs_shutdown(self, local_cfg):
        """A clean exit does not stop billing; the container restarts."""
        with attach(local_cfg, skip_gate=True, dry_run=True) as watch:
            ladder = watch.ladder
        assert any(r.verdict.probe == "failsafe.shutdown" for r in ladder.history)

    def test_an_exception_runs_shutdown_and_is_re_raised(self, local_cfg):
        captured = {}
        with pytest.raises(RuntimeError, match="CUDA OOM"):
            with attach(local_cfg, skip_gate=True, dry_run=True) as watch:
                captured["ladder"] = watch.ladder
                raise RuntimeError("CUDA OOM")

        shutdowns = [r for r in captured["ladder"].history
                     if r.verdict.probe == "failsafe.shutdown"]
        assert shutdowns and "crashed" in shutdowns[-1].verdict.detail

    def test_the_watchdog_thread_stops_on_exit(self, local_cfg):
        with attach(local_cfg, skip_gate=True, dry_run=True) as watch:
            assert watch._thread is not None
        assert watch._thread is None


class TestStartupGate:
    def test_a_failing_gate_raises_rather_than_training_on(self, local_cfg, monkeypatch):
        """The pod is already going away; proceeding would train on nothing."""
        monkeypatch.setenv("RUNPOD_API_KEY", "k")

        class NoGpus:
            def gpus(self):
                return []

            def torch_device_count(self):
                return 0

            def xid_errors(self):
                return []

            def nccl_all_reduce(self, timeout_s):
                return False

            def free_disk_gb(self, path):
                return 0.0

        import rlmwatch.probes.startup as startup

        monkeypatch.setattr(startup, "SystemHardware", lambda *a, **k: NoGpus())
        monkeypatch.setattr("rlmwatch.integration.SystemHardware", lambda *a, **k: NoGpus())

        with pytest.raises((StartupGateFailed, Exception)):
            with attach(local_cfg, hooks=StartupHooks(), dry_run=True):
                pass


def test_v0_train_hooks_demo_runs_end_to_end(monkeypatch, tmp_path):
    """The worked example must actually work, or it is documentation that lies."""
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-demo")
    monkeypatch.chdir(root)

    spec = importlib.util.spec_from_file_location(
        "train_hooks", root / "experiments" / "v0_smoke" / "train_hooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_hooks"] = module
    spec.loader.exec_module(module)

    assert module.main(["--demo", "--max-steps", "2"]) == 0

    # Without --demo it refuses rather than pretending to be a trainer.
    assert module.main([]) == 2
