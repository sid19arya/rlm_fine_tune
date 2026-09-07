"""In-memory fake of the W&B public API.

Models the parts of W&B the spec cares about, including the two behaviours that
make real W&B hard to monitor against:

* the run state and the training loop's progress are **independent** -- you can
  set `state="running"` while `_timestamp` goes stale, which is exactly the
  deadlocked-loop case that W&B's own heartbeat cannot see;
* the API can be made to raise, so probes can be checked for returning
  `unknown` rather than `fail` during a network outage.
"""

from __future__ import annotations

from typing import Any


class FakeApiError(RuntimeError):
    """Stands in for whatever wandb.Api raises when it cannot reach the server."""


class FakeRun:
    def __init__(
        self,
        path: str,
        *,
        state: str = "running",
        summary: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.path = path
        self.state = state
        self.summary: dict[str, Any] = dict(summary or {})
        self._history: list[dict[str, Any]] = list(history or [])

    def log(self, step: dict[str, Any], *, timestamp: float | None = None) -> None:
        """Append a history row and advance the progress timestamp.

        Callers that want a *stalled* run log nothing and let wall-clock move,
        or pass an old timestamp explicitly.
        """
        row = dict(step)
        self._history.append(row)
        self.summary.update(row)
        if timestamp is not None:
            self.summary["_timestamp"] = timestamp

    def history(self, keys: list[str] | None = None, samples: int = 500, pandas: bool = True):
        rows = self._history[-samples:]
        if keys is None:
            return [dict(r) for r in rows]
        return [{k: r[k] for k in keys if k in r} for r in rows]


class FakeWandbApi:
    """Drop-in for `wandb.Api()`; inject via `WandbClient(api=FakeWandbApi(...))`."""

    def __init__(self, runs: dict[str, FakeRun] | None = None) -> None:
        self.runs: dict[str, FakeRun] = dict(runs or {})
        #: Set to raise on every read, simulating a network partition.
        self.unavailable = False
        self.call_count = 0

    def add(self, run: FakeRun) -> FakeRun:
        self.runs[run.path] = run
        return run

    def run(self, path: str) -> FakeRun:
        self.call_count += 1
        if self.unavailable:
            raise FakeApiError("network is unreachable")
        try:
            return self.runs[path]
        except KeyError as exc:
            raise FakeApiError(f"run not found: {path}") from exc


def gpu_system_metrics(
    util_pct: float, *, gpus: int = 2, mem_allocated_pct: float = 70.0
) -> dict[str, float]:
    """Build the `system/gpu.N.*` summary keys W&B writes automatically."""
    out: dict[str, float] = {}
    for i in range(gpus):
        out[f"system/gpu.{i}.gpu"] = util_pct
        out[f"system/gpu.{i}.memoryAllocated"] = mem_allocated_pct
        out[f"system/gpu.{i}.temp"] = 62.0
        out[f"system/gpu.{i}.powerWatts"] = 250.0
    return out
