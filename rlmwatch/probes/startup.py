"""Startup gate (spec section 3).

Runs inside the pod, before the first training step. Its whole purpose is to
**convert slow expensive failures into fast cheap ones**: a misconfigured run
should die in ninety seconds, not six hours later when the first checkpoint
write fails on a full disk.

Two probes here earn their place more than the rest:

* the **checkpoint round-trip** -- actually writing a tensor to the real
  checkpoint directory, reading it back and deleting it. This one probe prevents
  the worst outcome in the whole system, a long run that cannot save.
* the **RunPod write-scope check** -- proving the key can write by writing.
  Discovering a read-only key at kill time defeats the entire failsafe layer,
  and every read call succeeds right up until that moment.

Everything hardware- or trainer-specific goes through an injectable backend, so
this module is testable with no GPU, no network and no model.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from rlmwatch.clients.runpod import RunPodAuthError
from rlmwatch.clients.wandb import WandbUnavailable
from rlmwatch.probes.base import BaseProbe, Context, Verdict, worst

log = logging.getLogger("rlmwatch.startup")


@dataclass
class GpuInfo:
    index: int
    name: str
    free_vram_gb: float
    total_vram_gb: float
    ecc_uncorrectable: int = 0
    ecc_correctable: int = 0
    retired_pages_pending: int = 0


class HardwareBackend(Protocol):
    """Everything the hardware probes need to ask the machine."""

    def gpus(self) -> list[GpuInfo]: ...
    def torch_device_count(self) -> int | None: ...
    def xid_errors(self) -> list[str]: ...
    def nccl_all_reduce(self, timeout_s: float) -> bool: ...
    def free_disk_gb(self, path: str, quota_gb: float | None = None) -> float | None: ...


class SystemHardware:
    """Real backend. Shells out to nvidia-smi; torch and NCCL are optional.

    nvidia-smi rather than torch for the inventory, so the gate still works
    before the training environment is fully importable -- which is exactly when
    a misconfigured pod most needs to be caught.
    """

    def __init__(self, runner=None) -> None:
        self._run = runner or self._default_run

    @staticmethod
    def _default_run(cmd: list[str], timeout: float = 30.0) -> str:
        if shutil.which(cmd[0]) is None:
            return ""
        try:
            result = subprocess.run(  # noqa: S603 - fixed diagnostic commands
                cmd, capture_output=True, text=True, timeout=timeout, check=False
            )
        except (subprocess.TimeoutExpired, OSError):
            return ""
        return result.stdout or ""

    def gpus(self) -> list[GpuInfo]:
        out = self._run([
            "nvidia-smi",
            "--query-gpu=index,name,memory.free,memory.total,"
            "ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.volatile.total,"
            "retired_pages.pending",
            "--format=csv,noheader,nounits",
        ])
        gpus: list[GpuInfo] = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue

            def number(value: str) -> float:
                try:
                    return float(value)
                except ValueError:
                    return 0.0  # "[N/A]" on GPUs without ECC reporting

            gpus.append(
                GpuInfo(
                    index=int(number(parts[0])),
                    name=parts[1],
                    free_vram_gb=number(parts[2]) / 1024.0,
                    total_vram_gb=number(parts[3]) / 1024.0,
                    ecc_uncorrectable=int(number(parts[4])) if len(parts) > 4 else 0,
                    ecc_correctable=int(number(parts[5])) if len(parts) > 5 else 0,
                    retired_pages_pending=int(number(parts[6])) if len(parts) > 6 else 0,
                )
            )
        return gpus

    def torch_device_count(self) -> int | None:
        try:
            import torch  # noqa: PLC0415 - optional, only present in the pod
        except ImportError:
            return None
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    def xid_errors(self) -> list[str]:
        out = self._run(["dmesg"], timeout=15.0)
        return [line for line in out.splitlines() if "Xid" in line]

    def nccl_all_reduce(self, timeout_s: float) -> bool:
        """Small all-reduce in a subprocess under a hard timeout.

        A subprocess, not a thread: a NCCL hang is uninterruptible from inside
        the process, and the gate must not become the indefinite block it exists
        to prevent.
        """
        script = (
            "import torch, torch.distributed as dist, os;"
            "dist.init_process_group('nccl', init_method='env://', world_size=1, rank=0);"
            "t = torch.ones(8, device='cuda');"
            "dist.all_reduce(t);"
            "assert t.sum().item() == 8;"
            "print('nccl-ok')"
        )
        env = {**os.environ, "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29517"}
        try:
            result = subprocess.run(  # noqa: S603
                ["python", "-c", script], capture_output=True, text=True,
                timeout=timeout_s, check=False, env=env,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        return "nccl-ok" in (result.stdout or "")

    def free_disk_gb(self, path: str, quota_gb: float | None = None) -> float | None:
        """Free space at `path`, honouring a volume quota when one is given.

        `shutil.disk_usage` reads statvfs, which on a network-backed volume
        reports the **cluster's** capacity rather than the share we are allowed
        to use. Observed on a RunPod 100GB volume backed by MooseFS:

            df:     756T total, 191T available
            actual: 91G of a 100G quota, writes already failing with
                    "Disk quota exceeded"

        The gate passed that disk as healthy while the run was minutes from
        dying on it -- the same shape as a budget cap that reads $0.00. So when
        `quota_gb` is configured, free space is computed as quota minus actual
        usage, and statvfs is used only as a ceiling for the local case.
        """
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            return None
        statvfs_free = usage.free / 1024**3
        if not quota_gb:
            return statvfs_free
        used = self._used_gb(path)
        if used is None:
            return statvfs_free
        # The smaller of the two: a quota can bind before the filesystem does,
        # and on a local disk the filesystem can bind before the quota.
        return min(statvfs_free, max(0.0, quota_gb - used))

    def _used_gb(self, path: str) -> float | None:
        """Actual bytes used under `path`. `du` is slow on a network mount but
        it is the only number that reflects a quota."""
        out = self._run(["du", "-sb", path], timeout=120.0)
        try:
            return int(out.split()[0]) / 1024**3
        except (ValueError, IndexError):
            return None


@dataclass
class StartupHooks:
    """Callables the training script lends the gate.

    Anything not supplied yields `unknown` rather than `ok`: an unrun check is
    never evidence of health, and the gate reports which checks were not wired.
    """

    load_model: Callable[[], dict[str, Any]] | None = None
    first_batch: Callable[[], dict[str, Any]] | None = None
    resume: Callable[[], int | None] | None = None
    train_step: Callable[[], float] | None = None
    rollout_cycle: Callable[[], float] | None = None
    gen_backend_health: Callable[[], bool] | None = None
    log_canary: Callable[[float], None] | None = None
    read_canary: Callable[[], float | None] | None = None


@dataclass
class StartupContext:
    """Context plus the extras only the startup gate needs."""

    ctx: Context
    hardware: HardwareBackend
    hooks: StartupHooks
    notifier: Any = None
    nccl_timeout_s: float = 120.0
    #: Phase durations measured during warm-up; `stall_threshold: auto` is
    #: resolved against these for the rest of the run.
    baselines: dict[str, float] = field(default_factory=dict)


class StartupProbe(BaseProbe):
    """Base for gate probes, which take the richer StartupContext."""

    def check(self, ctx: Context) -> Verdict:  # pragma: no cover - interface
        raise NotImplementedError("startup probes take a StartupContext; call inspect()")

    def inspect(self, sctx: StartupContext) -> Verdict:  # pragma: no cover - interface
        raise NotImplementedError


# --- hardware -----------------------------------------------------------------


class GpuInventoryProbe(StartupProbe):
    """Count, type and free VRAM against what the config expects."""

    name = "startup.gpu_inventory"

    def inspect(self, sctx: StartupContext) -> Verdict:
        expect = sctx.ctx.cfg.expect
        gpus = sctx.hardware.gpus()
        if not gpus:
            return self.fail("nvidia-smi reported no GPUs", expected=expect.gpu_count)

        names = [g.name for g in gpus]
        min_free = min(g.free_vram_gb for g in gpus)
        evidence = {"count": len(gpus), "expected_count": expect.gpu_count, "names": names,
                    "min_free_vram_gb": round(min_free, 1),
                    "required_free_vram_gb": expect.min_free_vram_gb}

        if len(gpus) != expect.gpu_count:
            return self.fail(
                f"expected {expect.gpu_count} GPUs, found {len(gpus)}", **evidence
            )
        if expect.gpu_type and not any(expect.gpu_type.lower() in n.lower() for n in names):
            return self.fail(
                f"expected {expect.gpu_type!r}, found {names}", **evidence
            )
        if min_free < expect.min_free_vram_gb:
            return self.fail(
                f"only {min_free:.1f}GB free VRAM, need {expect.min_free_vram_gb}GB -- "
                f"something else is already resident on this GPU", **evidence,
            )
        return self.ok(f"{len(gpus)}x {names[0]}, {min_free:.0f}GB free", **evidence)


class TorchCudaProbe(StartupProbe):
    """torch.cuda agrees with nvidia-smi.

    They disagree more often than expected -- CUDA_VISIBLE_DEVICES set by a
    template, a driver/runtime mismatch -- and the run then trains on half the
    GPUs it is paying for.
    """

    name = "startup.torch_cuda"

    def inspect(self, sctx: StartupContext) -> Verdict:
        count = sctx.hardware.torch_device_count()
        if count is None:
            return self.unknown("torch not importable in this environment")
        smi_count = len(sctx.hardware.gpus())
        evidence = {"torch_device_count": count, "nvidia_smi_count": smi_count}
        if count == 0:
            return self.fail("torch.cuda.is_available() is False", **evidence)
        if count != smi_count:
            return self.fail(
                f"torch sees {count} devices but nvidia-smi sees {smi_count} -- likely "
                f"CUDA_VISIBLE_DEVICES, and the run would use only part of what it pays "
                f"for", **evidence,
            )
        return self.ok(f"torch sees {count} CUDA devices", **evidence)


class GpuHealthProbe(StartupProbe):
    """ECC errors, pending retired pages and Xid events.

    Fail on Xid and uncorrectable ECC, warn on correctable. A bad GPU that is
    merely degraded will still finish the run and produce garbage.
    """

    name = "startup.gpu_health"

    def inspect(self, sctx: StartupContext) -> Verdict:
        gpus = sctx.hardware.gpus()
        xids = sctx.hardware.xid_errors()
        uncorrectable = sum(g.ecc_uncorrectable for g in gpus)
        correctable = sum(g.ecc_correctable for g in gpus)
        pending = sum(g.retired_pages_pending for g in gpus)
        evidence = {"xid_events": xids[:5], "ecc_uncorrectable": uncorrectable,
                    "ecc_correctable": correctable, "retired_pages_pending": pending}

        if xids:
            return self.fail(f"{len(xids)} Xid event(s) in dmesg -- bad GPU", **evidence)
        if uncorrectable:
            return self.fail(f"{uncorrectable} uncorrectable ECC error(s)", **evidence)
        if pending:
            return self.fail(f"{pending} retired page(s) pending", **evidence)
        if correctable:
            return self.warn(f"{correctable} correctable ECC error(s)", **evidence)
        return self.ok("no ECC or Xid errors", **evidence)


class NcclProbe(StartupProbe):
    """Multi-GPU all-reduce smoke test, under a hard timeout.

    A NCCL hang at startup is common and must not become an indefinite block --
    which is why the backend runs it in a subprocess it can actually kill.
    """

    name = "startup.nccl"

    def inspect(self, sctx: StartupContext) -> Verdict:
        if sctx.ctx.cfg.expect.gpu_count < 2:
            return self.ok("single GPU, no NCCL needed", gpu_count=1)
        started = time.monotonic()
        ok = sctx.hardware.nccl_all_reduce(sctx.nccl_timeout_s)
        elapsed = time.monotonic() - started
        evidence = {"timeout_s": sctx.nccl_timeout_s, "elapsed_s": round(elapsed, 1)}
        if not ok:
            return self.fail(
                "NCCL all-reduce failed or timed out -- multi-GPU training would hang "
                "here with no error", **evidence,
            )
        return self.ok(f"NCCL all-reduce passed in {elapsed:.1f}s", **evidence)


# --- storage ------------------------------------------------------------------


class DiskSpaceProbe(StartupProbe):
    """Free space at the checkpoint directory, sized against checkpoint x keep_n."""

    name = "startup.disk_space"

    def inspect(self, sctx: StartupContext) -> Verdict:
        expect = sctx.ctx.cfg.expect
        free = sctx.hardware.free_disk_gb(expect.checkpoint_dir, expect.volume_quota_gb)
        if free is None:
            return self.fail(
                f"checkpoint dir {expect.checkpoint_dir} does not exist or is not "
                f"readable", path=expect.checkpoint_dir,
            )
        evidence = {"free_gb": round(free, 1), "required_gb": expect.min_disk_gb,
                    "path": expect.checkpoint_dir}
        if free < expect.min_disk_gb:
            return self.fail(
                f"only {free:.0f}GB free at {expect.checkpoint_dir}, need "
                f"{expect.min_disk_gb}GB", **evidence,
            )
        return self.ok(f"{free:.0f}GB free at {expect.checkpoint_dir}", **evidence)


class CheckpointRoundTripProbe(StartupProbe):
    """Write, read back, delete -- in the *real* checkpoint directory.

    The single highest-value probe in the gate. It prevents the worst outcome
    the whole system can produce: a sixteen-hour run that cannot save. Read-only
    mounts, wrong ownership and full disks all pass a `df` check and fail here.
    """

    name = "startup.checkpoint_roundtrip"

    def inspect(self, sctx: StartupContext) -> Verdict:
        directory = sctx.ctx.cfg.expect.checkpoint_dir
        payload = b"rlmwatch-roundtrip-" + os.urandom(16)
        path = os.path.join(directory, ".rlmwatch-roundtrip")
        try:
            os.makedirs(directory, exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())  # a buffered write proves nothing
            with open(path, "rb") as handle:
                read_back = handle.read()
        except OSError as exc:
            return self.fail(
                f"cannot write to the checkpoint directory {directory}: {exc} -- this "
                f"run would train for hours and then fail to save",
                path=directory, error=str(exc),
            )
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        if read_back != payload:
            return self.fail(
                f"checkpoint round-trip returned different bytes at {directory} -- "
                f"the filesystem is not trustworthy", path=directory,
            )
        return self.ok(f"checkpoint round-trip passed at {directory}", path=directory)


# --- model and data -----------------------------------------------------------


class ModelLoadProbe(StartupProbe):
    """Model and tokenizer load; parameter count matches expectation."""

    name = "startup.model_load"

    def inspect(self, sctx: StartupContext) -> Verdict:
        if sctx.hooks.load_model is None:
            return self.unknown("no load_model hook wired")
        try:
            info = sctx.hooks.load_model()
        except Exception as exc:  # noqa: BLE001 - any load failure is a gate failure
            return self.fail(f"model failed to load: {exc}", error=str(exc))

        params = info.get("param_count")
        expected = info.get("expected_param_count")
        evidence = {k: v for k, v in info.items() if not k.startswith("_")}
        if expected and params and abs(params - expected) / expected > 0.01:
            return self.fail(
                f"loaded {params:,} parameters, expected ~{expected:,} -- wrong "
                f"checkpoint or wrong model", **evidence,
            )
        return self.ok(f"model loaded ({params:,} params)" if params else "model loaded",
                       **evidence)


class DataProbe(StartupProbe):
    """First batch materialises with the expected shapes and dtypes."""

    name = "startup.first_batch"

    def inspect(self, sctx: StartupContext) -> Verdict:
        if sctx.hooks.first_batch is None:
            return self.unknown("no first_batch hook wired")
        try:
            info = sctx.hooks.first_batch()
        except Exception as exc:  # noqa: BLE001
            return self.fail(f"first batch failed to materialise: {exc}", error=str(exc))

        max_seq_len = info.get("max_seq_len")
        longest = info.get("longest_sample")
        if max_seq_len and longest and longest > max_seq_len:
            return self.fail(
                f"a sample of {longest} tokens exceeds max_seq_len {max_seq_len} -- it "
                f"would be silently truncated", **info,
            )
        return self.ok("first batch materialised", **info)


class ResumeProbe(StartupProbe):
    """If a resume path is set, it loads and reports the step it resumed from."""

    name = "startup.resume"

    def inspect(self, sctx: StartupContext) -> Verdict:
        if sctx.hooks.resume is None:
            return self.unknown("no resume hook wired (fresh run)")
        try:
            step = sctx.hooks.resume()
        except Exception as exc:  # noqa: BLE001
            return self.fail(f"resume failed: {exc}", error=str(exc))
        if step is None:
            return self.ok("no checkpoint to resume from; starting fresh")
        return self.ok(f"resumed from step {step}", resumed_step=step)


# --- observability self-test --------------------------------------------------


class WandbVisibilityProbe(StartupProbe):
    """The run is readable from *outside* the process, and a canary round-trips.

    The monitor has to prove itself before the run trusts it. A W&B run that
    initialised locally but is not visible through the public API means the
    sentinel would be blind for the whole run and would report `unknown`
    forever.
    """

    name = "startup.wandb_visibility"

    def inspect(self, sctx: StartupContext) -> Verdict:
        ctx = sctx.ctx
        run_path = ctx.cfg.run.wandb
        if not run_path:
            return self.fail("run.wandb is not configured; the sentinel would be blind")
        try:
            state = ctx.wandb.state(run_path)
        except WandbUnavailable as exc:
            return self.fail(
                f"run {run_path} is not readable through the public API: {exc}",
                run=run_path,
            )

        if sctx.hooks.log_canary is None or sctx.hooks.read_canary is None:
            return self.ok(f"run {run_path} visible externally (state {state})",
                           run=run_path, state=state)

        canary = time.time()
        sctx.hooks.log_canary(canary)
        deadline = time.monotonic() + ctx.cfg.failsafe.confirm_delay_s
        while time.monotonic() < deadline:
            ctx.wandb.invalidate(run_path)
            if sctx.hooks.read_canary() == canary:
                return self.ok("canary metric round-tripped through W&B",
                               run=run_path, canary=canary)
            time.sleep(1.0)
        return self.fail(
            f"canary metric did not read back within {ctx.cfg.failsafe.confirm_delay_s}s "
            f"-- metrics are being written but not retrievable, so every liveness probe "
            f"would be blind", run=run_path,
        )


class RunPodWriteScopeProbe(StartupProbe):
    """Prove the API key can write, by writing.

    Every read succeeds with a read-only key. The failure surfaces at exactly
    the moment the failsafe tries to terminate an expensive pod, by which point
    the whole failsafe layer is decorative. Ninety seconds here instead.
    """

    name = "startup.runpod_write_scope"

    def inspect(self, sctx: StartupContext) -> Verdict:
        pod_id = sctx.ctx.cfg.run.pod_id
        if not pod_id:
            return self.fail(
                "run.pod_id is not set, so nothing can terminate this pod. Every "
                "failsafe in this library is inert without it."
            )
        try:
            sctx.ctx.runpod.verify_write_scope(pod_id)
        except RunPodAuthError as exc:
            return self.fail(
                f"the RunPod API key cannot write: {exc}. Reads work, so this would "
                f"have surfaced only at kill time, with the pod still billing.",
                pod_id=pod_id,
            )
        except Exception as exc:  # noqa: BLE001
            return self.unknown(f"could not verify write scope: {exc}", pod_id=pod_id)
        return self.ok("RunPod API key has write scope", pod_id=pod_id)


class NotifySinkProbe(StartupProbe):
    """Post a startup message. Its absence is itself the signal.

    A notification path exercised only during an incident is a notification path
    nobody knows is broken.
    """

    name = "startup.notify_sink"

    def inspect(self, sctx: StartupContext) -> Verdict:
        if sctx.notifier is None:
            return self.unknown("no notifier wired")
        if sctx.notifier.selftest():
            return self.ok("notification sink reachable")
        return self.fail(
            f"no notification sink accepted the startup message: "
            f"{sctx.notifier.last_error}"
        )


# --- warm-up ------------------------------------------------------------------


class TrainStepProbe(StartupProbe):
    """One full training step completes and logs. Records the duration."""

    name = "startup.train_step"

    def inspect(self, sctx: StartupContext) -> Verdict:
        if sctx.hooks.train_step is None:
            return self.unknown("no train_step hook wired")
        started = time.monotonic()
        try:
            loss = sctx.hooks.train_step()
        except Exception as exc:  # noqa: BLE001
            return self.fail(f"warm-up training step failed: {exc}", error=str(exc))
        elapsed = time.monotonic() - started
        sctx.baselines["step"] = elapsed
        sctx.baselines["throughput"] = 1.0 / elapsed if elapsed else 0.0
        return self.ok(f"warm-up step completed in {elapsed:.1f}s (loss {loss})",
                       duration_s=round(elapsed, 2), loss=loss)


class RolloutCycleProbe(StartupProbe):
    """RL only: one rollout -> score -> update cycle, timed.

    Its duration is the baseline every phase-aware staleness threshold is
    derived from, so a run without this measurement falls back to the `auto`
    floor rather than to a number somebody guessed.
    """

    name = "startup.rollout_cycle"

    def inspect(self, sctx: StartupContext) -> Verdict:
        if sctx.ctx.cfg.run.regime != "rl":
            return self.ok("not an RL run; no rollout cycle to measure")
        if sctx.hooks.rollout_cycle is None:
            return self.unknown("no rollout_cycle hook wired; stall thresholds will "
                                "fall back to the auto floor")
        started = time.monotonic()
        try:
            sctx.hooks.rollout_cycle()
        except Exception as exc:  # noqa: BLE001
            return self.fail(f"warm-up rollout cycle failed: {exc}", error=str(exc))
        elapsed = time.monotonic() - started
        sctx.baselines["rollout"] = elapsed
        return self.ok(
            f"rollout cycle completed in {elapsed:.1f}s; stall threshold for the rollout "
            f"phase is now {max(600.0, 4 * elapsed):.0f}s",
            duration_s=round(elapsed, 2),
        )


class GenBackendProbe(StartupProbe):
    """vLLM/SGLang health endpoint responds before training starts."""

    name = "startup.gen_backend"

    def inspect(self, sctx: StartupContext) -> Verdict:
        if sctx.hooks.gen_backend_health is None:
            return self.unknown("no gen_backend_health hook wired")
        try:
            healthy = sctx.hooks.gen_backend_health()
        except Exception as exc:  # noqa: BLE001
            return self.fail(f"generation backend health check raised: {exc}")
        if not healthy:
            return self.fail(
                "generation backend is not responding; the trainer would block on it "
                "indefinitely with no error"
            )
        return self.ok("generation backend healthy")


#: Order matters. Cheap and decisive first, expensive last: a wrong GPU count
#: should be caught before anything downloads a model, and the observability
#: self-test runs before warm-up so a failing gate can actually report itself.
GATE_PROBES: tuple[StartupProbe, ...] = (
    GpuInventoryProbe(),
    TorchCudaProbe(),
    GpuHealthProbe(),
    RunPodWriteScopeProbe(),
    DiskSpaceProbe(),
    CheckpointRoundTripProbe(),
    NcclProbe(),
    NotifySinkProbe(),
    WandbVisibilityProbe(),
    ModelLoadProbe(),
    DataProbe(),
    ResumeProbe(),
    TrainStepProbe(),
    RolloutCycleProbe(),
    GenBackendProbe(),
)


@dataclass
class GateResult:
    verdicts: list[Verdict]
    duration_s: float
    baselines: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(v.status == "fail" for v in self.verdicts)

    @property
    def failure(self) -> Verdict | None:
        return next((v for v in self.verdicts if v.status == "fail"), None)

    @property
    def unwired(self) -> list[str]:
        """Checks that could not run. Reported, never counted as passing."""
        return [v.probe for v in self.verdicts if v.status == "unknown"]

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for verdict in self.verdicts:
            counts[verdict.status] = counts.get(verdict.status, 0) + 1
        parts = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
        return f"startup gate {'PASSED' if self.passed else 'FAILED'} in " \
               f"{self.duration_s:.1f}s ({parts})"

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "duration_s": round(self.duration_s, 2),
            "baselines": self.baselines,
            "unwired": self.unwired,
            "verdicts": [v.as_dict() for v in self.verdicts],
        }


def gate(
    sctx: StartupContext,
    *,
    probes: tuple[StartupProbe, ...] = GATE_PROBES,
    on_failure: Callable[[GateResult], None] | None = None,
) -> GateResult:
    """Run the gate in order, short-circuiting on the first `fail`.

    Short-circuiting is deliberate: once one check has failed the pod is going
    away, and the remaining probes would only add noise and delay to a bill that
    is still accruing.

    The caller supplies `on_failure`, which is what actually terminates the pod.
    Keeping the kill out of this function is what lets the whole gate be tested
    without a pod to destroy -- but a failed gate must never be left running,
    and `startup.gate_and_enforce` is the wired-up version.
    """
    started = time.monotonic()
    verdicts: list[Verdict] = []
    for probe in probes:
        try:
            verdict = probe.inspect(sctx)
        except Exception as exc:  # noqa: BLE001 - a probe crash is a gate failure
            verdict = Verdict(
                probe=probe.name, status="fail",
                detail=f"probe raised {type(exc).__name__}: {exc}",
                evidence={"error": str(exc)},
            )
        verdicts.append(verdict)
        log.info("%s", verdict)
        if verdict.status == "fail":
            break

    result = GateResult(verdicts=verdicts, duration_s=time.monotonic() - started,
                        baselines=dict(sctx.baselines))
    if not result.passed and on_failure is not None:
        on_failure(result)
    return result


def gate_and_enforce(
    sctx: StartupContext,
    *,
    ladder: Any,
    probes: tuple[StartupProbe, ...] = GATE_PROBES,
    log_result: Callable[[dict[str, Any]], None] | None = None,
) -> GateResult:
    """The wired-up gate: run it, and never leave a failed pod running.

    On failure it emits diagnostics and then stops or terminates per
    `startup.on_failure`. On success it logs `startup_ok`, the gate duration and
    the measured baselines, which is what the sentinel watches for -- absence of
    `startup_ok` within `startup.deadline_s` of pod creation is itself a failure,
    and it is the only way to catch a crash that happens before `wandb.init()`,
    which W&B can never see.
    """

    def on_failure(result: GateResult) -> None:
        failure = result.failure
        assert failure is not None
        ladder.handle(failure)  # L0-L2: log, notify, snapshot
        # `startup.on_failure` is a separate setting from `failsafe.on_terminal`
        # because a gate failure happens before the first step: there are no
        # checkpoints to lose, so terminate is right here even when the
        # steady-state policy is `stop`. Passed explicitly rather than by
        # mutating the shared config.
        ladder.shutdown(
            f"startup gate failed: {failure.probe}",
            action=sctx.ctx.cfg.startup.on_failure,
        )

    result = gate(sctx, probes=probes, on_failure=on_failure)

    if result.passed and log_result is not None:
        payload = {"startup_ok": 1, "startup_duration_s": round(result.duration_s, 2)}
        payload.update({f"baseline/{k}": v for k, v in result.baselines.items()})
        log_result(payload)
    return result


__all__ = [
    "GATE_PROBES",
    "GateResult",
    "GpuInfo",
    "HardwareBackend",
    "StartupContext",
    "StartupHooks",
    "SystemHardware",
    "gate",
    "gate_and_enforce",
    "worst",
]
