"""W&B read client.

W&B already logs GPU utilisation, memory, temperature and power as system
metrics with no instrumentation, so this client only *reads*. The training
script writes through the normal `wandb.log` path.

The blind spot this client is built around: **W&B's heartbeat comes from its own
internal service process, not from your training loop.** A deadlocked loop keeps
heartbeating and the run shows `Running` forever. So run state alone is neither
necessary nor sufficient, and `last_progress_age` -- seconds since the newest
logged step -- is the signal everything else defers to.

The converse failure matters too: a long network outage can mark a live run
`Crashed`. That is why `confirm()` exists and why an unreachable API returns
`unknown` rather than `fail`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# W&B run states, per the spec.
RUN_STATES = ("running", "finished", "crashed", "failed", "killed", "pending")
TERMINAL_BAD_STATES = ("crashed", "failed", "killed")


class WandbUnavailable(RuntimeError):
    """The W&B API could not be reached or the run could not be read.

    Callers translate this into an `unknown` verdict, never a `fail`. Killing a
    healthy $30/hr pod because of a transient network blip is the exact failure
    mode the acceptance criteria forbid.
    """


@dataclass(frozen=True)
class SystemMetrics:
    """Per-device system metrics, averaged across devices where scalar."""

    gpu_util_pct: dict[int, float]
    gpu_mem_allocated_pct: dict[int, float]
    gpu_temp_c: dict[int, float]
    gpu_power_w: dict[int, float]

    def mean_util(self) -> float | None:
        if not self.gpu_util_pct:
            return None
        return sum(self.gpu_util_pct.values()) / len(self.gpu_util_pct)

    def min_free_mem_pct(self) -> float | None:
        if not self.gpu_mem_allocated_pct:
            return None
        return 100.0 - max(self.gpu_mem_allocated_pct.values())


class WandbClient:
    """Read-only view of one W&B run.

    `wandb` is an optional dependency: the sentinel must be installable on a
    laptop, and the fault-injection suite runs against a fake. The import is
    therefore deferred to first use.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        api: Any = None,
        now: Callable[[], float] = time.time,
        cache_ttl_s: float = 5.0,
    ) -> None:
        self._api_key = api_key
        self._api = api
        self._now = now
        self._cache_ttl_s = cache_ttl_s
        self._cache: dict[str, tuple[float, Any]] = {}

    # --- plumbing ------------------------------------------------------------

    @property
    def api(self) -> Any:
        if self._api is None:
            try:
                import wandb  # noqa: PLC0415 - optional dependency, deferred on purpose
            except ImportError as exc:  # pragma: no cover - depends on env
                raise WandbUnavailable(
                    "wandb is not installed. Install the 'wandb' extra in the pod, or "
                    "inject a client via WandbClient(api=...)."
                ) from exc
            self._api = wandb.Api(api_key=self._api_key)
        return self._api

    def _run(self, run_path: str) -> Any:
        """Fetch a run, with a short TTL cache.

        Several probes run in the same sweep and each wants the same run
        object; without the cache a single sentinel tick would issue a dozen
        identical API calls, and the whole sentinel has to stay under $1/day.
        """
        cached = self._cache.get(run_path)
        now = self._now()
        if cached and now - cached[0] < self._cache_ttl_s:
            return cached[1]
        try:
            run = self.api.run(run_path)
        except Exception as exc:  # noqa: BLE001 - wandb raises a wide variety
            raise WandbUnavailable(f"could not read run {run_path}: {exc}") from exc
        self._cache[run_path] = (now, run)
        return run

    def invalidate(self, run_path: str | None = None) -> None:
        """Drop cached run objects. Required before a confirmation re-poll."""
        if run_path is None:
            self._cache.clear()
        else:
            self._cache.pop(run_path, None)

    # --- reads ---------------------------------------------------------------

    def state(self, run_path: str) -> str:
        """Run state, lower-cased: running / finished / crashed / failed / killed / pending."""
        return str(self._run(run_path).state or "unknown").lower()

    def last_progress_age(self, run_path: str) -> float:
        """Seconds since the newest logged step.

        **The single most important signal in this library.** Unlike run state,
        it comes from the training loop itself: if the loop is wedged, nothing
        new is logged and this number grows, no matter how healthily wandb's
        service process keeps heartbeating.
        """
        run = self._run(run_path)
        summary = getattr(run, "summary", None) or {}
        timestamp = None
        for key in ("_timestamp", "timestamp"):
            try:
                value = summary[key]
            except (KeyError, TypeError):
                value = None
            if value is not None:
                timestamp = float(value)
                break
        if timestamp is None:
            raise WandbUnavailable(
                f"run {run_path} has no _timestamp in its summary -- nothing has been "
                f"logged yet, or the run was never initialised"
            )
        return max(0.0, self._now() - timestamp)

    def summary(self, run_path: str) -> dict[str, Any]:
        run = self._run(run_path)
        raw = getattr(run, "summary", None) or {}
        try:
            return dict(raw)
        except (TypeError, ValueError):
            return {k: raw[k] for k in getattr(raw, "keys", list)()}

    def metric_window(self, run_path: str, key: str, n: int = 50) -> list[float]:
        """The most recent `n` finite values of `key`, oldest first.

        Non-finite and missing samples are dropped rather than coerced: a probe
        asking "is the trend falling" must not be handed a zero that the trainer
        never logged.
        """
        run = self._run(run_path)
        try:
            history = run.history(keys=[key], samples=max(n * 2, n + 10), pandas=False)
        except Exception as exc:  # noqa: BLE001 - wandb raises broadly
            raise WandbUnavailable(f"could not read history {key} for {run_path}: {exc}") from exc

        values: list[float] = []
        for row in history or []:
            value = row.get(key) if isinstance(row, dict) else None
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number != number or number in (float("inf"), float("-inf")):
                # NaN/Inf are meaningful to the loss probe, which reads the raw
                # summary; trend probes must not average them in.
                continue
            values.append(number)
        return values[-n:]

    def raw_metric_window(self, run_path: str, key: str, n: int = 50) -> list[float]:
        """Like `metric_window`, but keeps NaN/Inf so the loss probe can see them."""
        run = self._run(run_path)
        try:
            history = run.history(keys=[key], samples=max(n * 2, n + 10), pandas=False)
        except Exception as exc:  # noqa: BLE001
            raise WandbUnavailable(f"could not read history {key} for {run_path}: {exc}") from exc
        values: list[float] = []
        for row in history or []:
            value = row.get(key) if isinstance(row, dict) else None
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        return values[-n:]

    def system_metrics(self, run_path: str) -> SystemMetrics:
        """GPU utilisation, memory, temperature and power from W&B's own agent.

        These need no instrumentation in the training script, which is why they
        are the sentinel's cheapest liveness cross-check: 0% utilisation while
        the run claims to be `Running` is a wedged process.
        """
        summary = self.summary(run_path)
        util: dict[int, float] = {}
        mem: dict[int, float] = {}
        temp: dict[int, float] = {}
        power: dict[int, float] = {}

        for key, value in summary.items():
            if not isinstance(key, str) or not key.startswith("system/gpu."):
                continue
            parts = key.split(".")
            if len(parts) < 3:
                continue
            try:
                index = int(parts[1])
                number = float(value)
            except (TypeError, ValueError):
                continue
            field = parts[-1]
            if field == "gpu":
                util[index] = number
            elif field == "memoryAllocated":
                mem[index] = number
            elif field == "temp":
                temp[index] = number
            elif field == "powerWatts":
                power[index] = number

        return SystemMetrics(
            gpu_util_pct=util,
            gpu_mem_allocated_pct=mem,
            gpu_temp_c=temp,
            gpu_power_w=power,
        )

    def confirm(self, run_path: str, predicate: Callable[[], bool], *, delay_s: float,
                sleep: Callable[[float], None] = time.sleep) -> bool:
        """Re-evaluate `predicate` after `delay_s` with a cold cache.

        Never trust a single poll. Every `fail` that leads to an action passes
        through here first, because a network blip must not terminate a healthy
        pod. Returns True only if the predicate holds on both readings.
        """
        if not predicate():
            return False
        sleep(delay_s)
        self.invalidate(run_path)
        return predicate()
