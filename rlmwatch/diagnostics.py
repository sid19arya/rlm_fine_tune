"""L2 snapshot bundle.

The point of this module, in one line from the spec: **a hung run that is killed
without a py-spy dump has taught you nothing and will hang again.** By the time
the pod is terminated the stacks are gone forever, so the bundle is assembled
before anything in the ladder is allowed to kill anything.

Every collector is individually guarded. A snapshot taken during an incident is
being taken on a machine that is already misbehaving -- `nvidia-smi` may hang,
`dmesg` may be unreadable in the container, the W&B API may be the thing that
broke. One failing collector records its error and the rest of the bundle still
gets written.

Log collection is bounded (500 lines by default). Unbounded log streaming into
an agent context is an explicit non-goal.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("rlmwatch.diagnostics")

DEFAULT_LOG_LINES = 500
DEFAULT_METRIC_STEPS = 50

#: Metrics worth capturing in a snapshot regardless of regime.
SNAPSHOT_METRICS = (
    "train/loss",
    "train/grad_norm",
    "reward/mean",
    "reward/std",
    "kl/ref",
    "policy/entropy",
    "completion/length",
    "format/success_rate",
    "rollout/duration_s",
)


def _run(cmd: list[str], *, timeout: float = 20.0) -> str:
    """Run a diagnostic command with a hard timeout.

    The timeout is not optional: `nvidia-smi` genuinely hangs on a wedged
    driver, and a diagnostics call that blocks forever prevents the very
    termination it was collected for.
    """
    if shutil.which(cmd[0]) is None:
        return f"<{cmd[0]} not installed>"
    try:
        result = subprocess.run(  # noqa: S603 - fixed diagnostic commands, no shell
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return f"<{' '.join(cmd)} timed out after {timeout}s>"
    except OSError as exc:
        return f"<{' '.join(cmd)} failed: {exc}>"
    return (result.stdout or "") + (result.stderr or "")


def python_pids(exclude_self: bool = True) -> list[int]:
    """PIDs of Python processes on this host.

    Used to find the trainer and the generation backend without being told
    which they are -- during an incident the caller often does not know.
    """
    pids: list[int] = []
    if sys.platform == "win32":  # pragma: no cover - pods are Linux
        return pids
    proc = Path("/proc")
    if not proc.is_dir():
        return pids
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if exclude_self and pid == os.getpid():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if "python" in cmdline:
            pids.append(pid)
    return pids


@dataclass
class Snapshot:
    """A diagnostic bundle. Serialisable, and small enough to attach to an alert."""

    reason: str
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stacks: dict[str, str] = field(default_factory=dict)
    gpu: str = ""
    logs: list[str] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(default_factory=dict)
    dmesg: str = ""
    disk: str = ""
    errors: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "at": self.at.isoformat(),
            "stacks": self.stacks,
            "gpu": self.gpu,
            "logs": self.logs,
            "metrics": self.metrics,
            "dmesg": self.dmesg,
            "disk": self.disk,
            "errors": self.errors,
        }

    def write(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.at.strftime("%Y%m%dT%H%M%SZ")
        path = directory / f"snapshot-{stamp}.json"
        path.write_text(json.dumps(self.as_dict(), indent=2, default=str), encoding="utf-8")
        return path

    def summary(self) -> str:
        parts = [f"snapshot({self.reason})"]
        if self.stacks:
            parts.append(f"{len(self.stacks)} stack dump(s)")
        if self.logs:
            parts.append(f"{len(self.logs)} log lines")
        if self.metrics:
            parts.append(f"{len(self.metrics)} metric series")
        if self.errors:
            parts.append(f"{len(self.errors)} collector error(s)")
        return ", ".join(parts)


class Diagnostics:
    """Assembles snapshots. Collectors are injectable so tests need no GPU."""

    def __init__(
        self,
        cfg: Any,
        *,
        runpod: Any = None,
        wandb: Any = None,
        runner=_run,
        pid_finder=python_pids,
    ) -> None:
        self.cfg = cfg
        self.runpod = runpod
        self.wandb = wandb
        self._run = runner
        self._pids = pid_finder

    def _guard(self, snapshot: Snapshot, name: str, fn):
        """Run one collector; record its failure rather than losing the bundle."""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - the host is already misbehaving
            snapshot.errors[name] = f"{type(exc).__name__}: {exc}"
            log.warning("diagnostic collector %s failed: %s", name, exc)
            return None

    def py_spy_dump(self, pid: int) -> str:
        """`--nonblocking` is deliberate: attaching normally can stop a process
        that is still limping along, and freezing a hung trainer mid-dump has
        cost people the checkpoint they were about to save."""
        return self._run(["py-spy", "dump", "--pid", str(pid), "--nonblocking"])

    def snapshot(
        self,
        reason: str,
        *,
        pids: list[int] | None = None,
        log_lines: int = DEFAULT_LOG_LINES,
        metric_steps: int = DEFAULT_METRIC_STEPS,
        metrics: tuple[str, ...] = SNAPSHOT_METRICS,
    ) -> Snapshot:
        """Assemble the full bundle. Never raises."""
        snap = Snapshot(reason=reason)

        # Stack traces first: they are the most perishable and the most useful.
        # In the RL case this covers the generation backend too, which is why
        # every Python process is dumped rather than just the trainer.
        targets = pids if pids is not None else self._guard(snap, "pids", self._pids) or []
        for pid in targets:
            dump = self._guard(snap, f"py-spy:{pid}", lambda p=pid: self.py_spy_dump(p))
            if dump:
                snap.stacks[str(pid)] = dump

        snap.gpu = self._guard(snap, "nvidia-smi",
                               lambda: self._run(["nvidia-smi", "-q"])) or ""
        snap.dmesg = self._guard(  # Xid and OOM-killer evidence
            snap, "dmesg", lambda: self._tail(self._run(["dmesg"]), 100)) or ""
        snap.disk = self._guard(snap, "df", lambda: self._run(["df", "-h"])) or ""

        if self.runpod and self.cfg.run.pod_id:
            lines = self._guard(
                snap, "runpod-logs",
                lambda: self.runpod.tail_lines(self.cfg.run.pod_id, n=log_lines),
            )
            if lines:
                snap.logs = [f"[{entry.source}] {entry.line}" for entry in lines]

        if self.wandb and self.cfg.run.wandb:
            for canonical in metrics:
                key = self.cfg.health.key_for(canonical)
                values = self._guard(
                    snap, f"metric:{key}",
                    lambda k=key: self.wandb.metric_window(self.cfg.run.wandb, k, metric_steps),
                )
                if values:
                    snap.metrics[key] = values

        return snap

    @staticmethod
    def _tail(text: str, n: int) -> str:
        lines = text.splitlines()
        return "\n".join(lines[-n:])

    def upload(self, snapshot: Snapshot, *, wandb_module: Any = None) -> str | None:
        """Attach the bundle to the W&B run as an artifact.

        Best-effort by design: if W&B is what broke, the snapshot has already
        been written to disk by the caller and is not lost.
        """
        module = wandb_module
        if module is None:
            try:
                import wandb as module  # noqa: PLC0415 - optional dependency
            except ImportError:
                return None
        try:
            if getattr(module, "run", None) is None:
                return None
            artifact = module.Artifact(
                name=f"rlmwatch-snapshot-{snapshot.at.strftime('%Y%m%dT%H%M%SZ')}",
                type="diagnostics",
            )
            with artifact.new_file("snapshot.json", mode="w") as handle:
                handle.write(json.dumps(snapshot.as_dict(), indent=2, default=str))
            module.run.log_artifact(artifact)
            return artifact.name
        except Exception as exc:  # noqa: BLE001
            log.warning("snapshot artifact upload failed: %s", exc)
            return None
