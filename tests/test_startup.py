"""Startup gate (spec section 3).

The bar: a misconfigured run dies in ninety seconds, not six hours in.
"""

from __future__ import annotations

import pytest

from rlmwatch.actions import EscalationLadder
from rlmwatch.notify import Notifier, RecordingSink
from rlmwatch.probes.startup import (
    GATE_PROBES,
    CheckpointRoundTripProbe,
    DataProbe,
    DiskSpaceProbe,
    GenBackendProbe,
    GpuHealthProbe,
    GpuInfo,
    GpuInventoryProbe,
    ModelLoadProbe,
    NcclProbe,
    NotifySinkProbe,
    ResumeProbe,
    RolloutCycleProbe,
    RunPodWriteScopeProbe,
    StartupContext,
    StartupHooks,
    TorchCudaProbe,
    TrainStepProbe,
    WandbVisibilityProbe,
    gate,
    gate_and_enforce,
)
from tests.conftest import POD_ID


class FakeHardware:
    """Two healthy A40s unless a test says otherwise."""

    def __init__(self, *, gpus=None, torch_count=2, xids=None, nccl=True, free_disk=200.0):
        self._gpus = gpus if gpus is not None else [
            GpuInfo(i, "NVIDIA A40", free_vram_gb=47.0, total_vram_gb=48.0) for i in range(2)
        ]
        self._torch_count = torch_count
        self._xids = xids or []
        self._nccl = nccl
        self._free_disk = free_disk
        self.nccl_timeout_seen = None

    def gpus(self):
        return self._gpus

    def torch_device_count(self):
        return self._torch_count

    def xid_errors(self):
        return self._xids

    def nccl_all_reduce(self, timeout_s):
        self.nccl_timeout_seen = timeout_s
        return self._nccl

    def free_disk_gb(self, path):
        return self._free_disk


@pytest.fixture
def hardware():
    return FakeHardware()


@pytest.fixture
def sink():
    return RecordingSink()


@pytest.fixture
def make_sctx(make_ctx, cfg, sink, tmp_path):
    def _make(*, hardware=None, hooks=None, notifier=None, ctx=None, **ctx_kwargs):
        return StartupContext(
            ctx=ctx or make_ctx(**ctx_kwargs),
            hardware=hardware or FakeHardware(),
            hooks=hooks or StartupHooks(),
            notifier=notifier if notifier is not None else Notifier(cfg, sinks=[sink]),
        )

    return _make


class TestHardware:
    def test_matching_inventory_passes(self, make_sctx):
        assert GpuInventoryProbe().inspect(make_sctx()).status == "ok"

    def test_wrong_gpu_count_fails(self, make_sctx):
        hw = FakeHardware(gpus=[GpuInfo(0, "NVIDIA A40", 47.0, 48.0)])
        verdict = GpuInventoryProbe().inspect(make_sctx(hardware=hw))
        assert verdict.status == "fail"
        assert verdict.evidence["count"] == 1

    def test_wrong_gpu_type_fails(self, make_sctx):
        hw = FakeHardware(gpus=[GpuInfo(i, "NVIDIA A100 80GB", 79.0, 80.0) for i in range(2)])
        assert GpuInventoryProbe().inspect(make_sctx(hardware=hw)).status == "fail"

    def test_occupied_vram_fails_before_the_run_starts(self, make_sctx):
        """Something already resident means OOM later, not now."""
        hw = FakeHardware(gpus=[GpuInfo(i, "NVIDIA A40", 3.0, 48.0) for i in range(2)])
        verdict = GpuInventoryProbe().inspect(make_sctx(hardware=hw))
        assert verdict.status == "fail"
        assert "already resident" in verdict.detail

    def test_torch_disagreeing_with_nvidia_smi_fails(self, make_sctx):
        """Usually CUDA_VISIBLE_DEVICES; the run pays for GPUs it never uses."""
        verdict = TorchCudaProbe().inspect(make_sctx(hardware=FakeHardware(torch_count=1)))
        assert verdict.status == "fail"
        assert "CUDA_VISIBLE_DEVICES" in verdict.detail

    def test_torch_absent_is_unknown_not_fail(self, make_sctx):
        hw = FakeHardware(torch_count=None)
        assert TorchCudaProbe().inspect(make_sctx(hardware=hw)).status == "unknown"

    def test_xid_events_fail(self, make_sctx):
        hw = FakeHardware(xids=["NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus"])
        assert GpuHealthProbe().inspect(make_sctx(hardware=hw)).status == "fail"

    def test_uncorrectable_ecc_fails_but_correctable_only_warns(self, make_sctx):
        bad = FakeHardware(gpus=[GpuInfo(0, "NVIDIA A40", 47.0, 48.0, ecc_uncorrectable=3)])
        assert GpuHealthProbe().inspect(make_sctx(hardware=bad)).status == "fail"
        soft = FakeHardware(gpus=[GpuInfo(0, "NVIDIA A40", 47.0, 48.0, ecc_correctable=2)])
        assert GpuHealthProbe().inspect(make_sctx(hardware=soft)).status == "warn"

    def test_nccl_failure_fails_under_a_hard_timeout(self, make_sctx):
        hw = FakeHardware(nccl=False)
        verdict = NcclProbe().inspect(make_sctx(hardware=hw))
        assert verdict.status == "fail"
        assert hw.nccl_timeout_seen == 120.0
        assert "would hang here" in verdict.detail

    def test_single_gpu_skips_nccl(self, make_sctx, config_dict):
        from rlmwatch.config import from_dict

        config_dict["expect"] = {**config_dict["expect"], "gpu_count": 1}
        sctx = make_sctx(hardware=FakeHardware(nccl=False))
        sctx.ctx.cfg = from_dict(config_dict)
        assert NcclProbe().inspect(sctx).status == "ok"


class TestStorage:
    def test_sufficient_disk_passes(self, make_sctx, tmp_path, config_dict):
        from rlmwatch.config import from_dict

        config_dict["expect"] = {**config_dict["expect"], "checkpoint_dir": str(tmp_path)}
        sctx = make_sctx()
        sctx.ctx.cfg = from_dict(config_dict)
        assert DiskSpaceProbe().inspect(sctx).status == "ok"

    def test_insufficient_disk_fails(self, make_sctx):
        assert DiskSpaceProbe().inspect(make_sctx(hardware=FakeHardware(free_disk=1.0))
                                        ).status == "fail"

    def test_missing_checkpoint_dir_fails(self, make_sctx):
        hw = FakeHardware(free_disk=None)
        assert DiskSpaceProbe().inspect(make_sctx(hardware=hw)).status == "fail"

    def test_checkpoint_roundtrip_passes_on_a_writable_dir(self, make_sctx, tmp_path,
                                                           config_dict):
        from rlmwatch.config import from_dict

        config_dict["expect"] = {**config_dict["expect"],
                                 "checkpoint_dir": str(tmp_path / "ckpt")}
        sctx = make_sctx()
        sctx.ctx.cfg = from_dict(config_dict)
        verdict = CheckpointRoundTripProbe().inspect(sctx)
        assert verdict.status == "ok"
        assert not list((tmp_path / "ckpt").glob(".rlmwatch*")), "probe must clean up"

    def test_unwritable_checkpoint_dir_fails_before_hours_are_spent(self, make_sctx,
                                                                    config_dict, tmp_path):
        """Prevents the worst outcome in the system: a run that cannot save."""
        from rlmwatch.config import from_dict

        blocker = tmp_path / "not-a-dir"
        blocker.write_text("I am a file", encoding="utf-8")
        config_dict["expect"] = {**config_dict["expect"],
                                 "checkpoint_dir": str(blocker / "ckpt")}
        sctx = make_sctx()
        sctx.ctx.cfg = from_dict(config_dict)
        verdict = CheckpointRoundTripProbe().inspect(sctx)
        assert verdict.status == "fail"
        assert "fail to save" in verdict.detail


class TestObservabilitySelfTest:
    def test_read_only_key_fails_the_gate(self, make_sctx, runpod_server):
        """Every read works; the failure would otherwise surface at kill time."""
        runpod_server.read_only_key = True
        verdict = RunPodWriteScopeProbe().inspect(make_sctx())
        assert verdict.status == "fail"
        assert "kill time" in verdict.detail

    def test_write_capable_key_passes(self, make_sctx):
        assert RunPodWriteScopeProbe().inspect(make_sctx()).status == "ok"

    def test_missing_pod_id_fails_because_nothing_could_terminate(self, make_sctx,
                                                                   config_dict):
        from rlmwatch.config import from_dict

        config_dict["run"] = {**config_dict["run"], "pod_id": ""}
        sctx = make_sctx()
        sctx.ctx.cfg = from_dict(config_dict)
        verdict = RunPodWriteScopeProbe().inspect(sctx)
        assert verdict.status == "fail"
        assert "inert" in verdict.detail

    def test_wandb_run_visible_externally_passes(self, make_sctx):
        assert WandbVisibilityProbe().inspect(make_sctx()).status == "ok"

    def test_wandb_run_not_readable_fails(self, make_sctx, wandb_api):
        wandb_api.unavailable = True
        verdict = WandbVisibilityProbe().inspect(make_sctx())
        assert verdict.status == "fail"

    def test_canary_round_trip(self, make_sctx):
        stored = {}
        hooks = StartupHooks(log_canary=lambda v: stored.update(v=v),
                             read_canary=lambda: stored.get("v"))
        verdict = WandbVisibilityProbe().inspect(make_sctx(hooks=hooks))
        assert verdict.status == "ok"
        assert "round-tripped" in verdict.detail

    def test_notify_sink_selftest_passes(self, make_sctx, sink):
        assert NotifySinkProbe().inspect(make_sctx()).status == "ok"
        assert sink.alerts[0].verdict.probe == "notify.selftest"

    def test_unreachable_notify_sink_fails(self, make_sctx, cfg):
        class BrokenSink:
            name = "broken"

            def send(self, alert):
                raise RuntimeError("slack down")

        notifier = Notifier(cfg, sinks=[BrokenSink()])
        assert NotifySinkProbe().inspect(make_sctx(notifier=notifier)).status == "fail"


class TestModelDataAndWarmup:
    def test_model_load_reports_param_count(self, make_sctx):
        hooks = StartupHooks(load_model=lambda: {"param_count": 8_000_000_000})
        verdict = ModelLoadProbe().inspect(make_sctx(hooks=hooks))
        assert verdict.status == "ok"
        assert "8,000,000,000" in verdict.detail

    def test_wrong_param_count_fails(self, make_sctx):
        hooks = StartupHooks(
            load_model=lambda: {"param_count": 500_000_000,
                                "expected_param_count": 8_000_000_000}
        )
        verdict = ModelLoadProbe().inspect(make_sctx(hooks=hooks))
        assert verdict.status == "fail"
        assert "wrong checkpoint" in verdict.detail

    def test_model_load_exception_is_a_gate_failure(self, make_sctx):
        def boom():
            raise RuntimeError("no such repo")

        assert ModelLoadProbe().inspect(make_sctx(hooks=StartupHooks(load_model=boom))
                                        ).status == "fail"

    def test_oversized_sample_fails_before_silent_truncation(self, make_sctx):
        hooks = StartupHooks(
            first_batch=lambda: {"max_seq_len": 4096, "longest_sample": 9000}
        )
        verdict = DataProbe().inspect(make_sctx(hooks=hooks))
        assert verdict.status == "fail"
        assert "silently truncated" in verdict.detail

    def test_resume_reports_the_step(self, make_sctx):
        hooks = StartupHooks(resume=lambda: 25)
        verdict = ResumeProbe().inspect(make_sctx(hooks=hooks))
        assert verdict.evidence["resumed_step"] == 25

    def test_warmup_step_records_a_baseline(self, make_sctx):
        hooks = StartupHooks(train_step=lambda: 1.23)
        sctx = make_sctx(hooks=hooks)
        assert TrainStepProbe().inspect(sctx).status == "ok"
        assert "step" in sctx.baselines and "throughput" in sctx.baselines

    def test_rollout_cycle_sets_the_phase_baseline(self, make_sctx):
        hooks = StartupHooks(rollout_cycle=lambda: None)
        sctx = make_sctx(hooks=hooks)
        verdict = RolloutCycleProbe().inspect(sctx)
        assert verdict.status == "ok"
        assert "rollout" in sctx.baselines
        assert "stall threshold" in verdict.detail

    def test_unwired_rollout_hook_is_unknown_and_says_what_it_costs(self, make_sctx):
        verdict = RolloutCycleProbe().inspect(make_sctx())
        assert verdict.status == "unknown"
        assert "auto floor" in verdict.detail

    def test_dead_generation_backend_fails(self, make_sctx):
        hooks = StartupHooks(gen_backend_health=lambda: False)
        verdict = GenBackendProbe().inspect(make_sctx(hooks=hooks))
        assert verdict.status == "fail"
        assert "indefinitely" in verdict.detail


class TestGate:
    def test_healthy_pod_passes_the_full_gate(self, make_sctx, tmp_path, config_dict):
        from rlmwatch.config import from_dict

        config_dict["expect"] = {**config_dict["expect"],
                                 "checkpoint_dir": str(tmp_path / "ckpt")}
        sctx = make_sctx(hooks=StartupHooks(
            load_model=lambda: {"param_count": 8_000_000_000},
            first_batch=lambda: {"max_seq_len": 4096, "longest_sample": 2000},
            train_step=lambda: 1.0,
            rollout_cycle=lambda: None,
            gen_backend_health=lambda: True,
        ))
        sctx.ctx.cfg = from_dict(config_dict)
        result = gate(sctx)
        assert result.passed, result.failure
        assert result.baselines["rollout"] >= 0

    def test_gate_short_circuits_on_the_first_failure(self, make_sctx):
        """Once the pod is going away, further probes only add delay to the bill."""
        sctx = make_sctx(hardware=FakeHardware(gpus=[]))
        result = gate(sctx)
        assert not result.passed
        assert len(result.verdicts) == 1
        assert result.failure.probe == "startup.gpu_inventory"

    def test_a_probe_that_raises_fails_the_gate(self, make_sctx):
        class Exploding:
            name = "startup.exploding"

            def inspect(self, sctx):
                raise RuntimeError("boom")

        result = gate(make_sctx(), probes=(Exploding(),))
        assert not result.passed
        assert "boom" in result.failure.detail

    def test_unwired_checks_are_reported_never_counted_as_passing(self, make_sctx,
                                                                   tmp_path, config_dict):
        from rlmwatch.config import from_dict

        config_dict["expect"] = {**config_dict["expect"],
                                 "checkpoint_dir": str(tmp_path / "ckpt")}
        sctx = make_sctx()
        sctx.ctx.cfg = from_dict(config_dict)
        result = gate(sctx)
        assert result.passed
        assert "startup.model_load" in result.unwired
        assert "startup.train_step" in result.unwired

    def test_a_failed_gate_never_leaves_the_pod_running(self, make_sctx, cfg, sink,
                                                        runpod_client, runpod_server):
        ladder = EscalationLadder(cfg, notifier=Notifier(cfg, sinks=[sink]),
                                  runpod=runpod_client, sleep=lambda _: None)
        sctx = make_sctx(hardware=FakeHardware(gpus=[]))
        result = gate_and_enforce(sctx, ladder=ladder)
        assert not result.passed
        # startup.on_failure defaults to terminate, and overrides the
        # steady-state `stop` policy: nothing has been checkpointed yet.
        assert runpod_server.terminated == [POD_ID]
        assert ladder.cfg.failsafe.on_terminal == "stop", "shared config must not be mutated" 

    def test_a_passing_gate_logs_startup_ok_and_the_baselines(self, make_sctx, cfg, sink,
                                                              runpod_client, tmp_path,
                                                              config_dict):
        from rlmwatch.config import from_dict

        config_dict["expect"] = {**config_dict["expect"],
                                 "checkpoint_dir": str(tmp_path / "ckpt")}
        logged = {}
        ladder = EscalationLadder(cfg, notifier=Notifier(cfg, sinks=[sink]),
                                  runpod=runpod_client, sleep=lambda _: None)
        sctx = make_sctx(hooks=StartupHooks(train_step=lambda: 1.0,
                                            rollout_cycle=lambda: None))
        sctx.ctx.cfg = from_dict(config_dict)
        result = gate_and_enforce(sctx, ladder=ladder, log_result=logged.update)
        assert result.passed
        assert logged["startup_ok"] == 1
        assert "startup_duration_s" in logged
        assert "baseline/rollout" in logged

    def test_gate_order_puts_cheap_decisive_checks_first(self):
        """A wrong GPU count should be caught before anything downloads a model."""
        names = [p.name for p in GATE_PROBES]
        assert names.index("startup.gpu_inventory") < names.index("startup.model_load")
        assert names.index("startup.runpod_write_scope") < names.index("startup.train_step")
        assert names.index("startup.notify_sink") < names.index("startup.train_step")

    def test_result_summary_is_one_line(self, make_sctx):
        result = gate(make_sctx(hardware=FakeHardware(gpus=[])))
        assert "FAILED" in result.summary()
        assert "\n" not in result.summary()
