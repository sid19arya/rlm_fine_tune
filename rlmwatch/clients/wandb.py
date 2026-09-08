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

    def latest_step(self, run_path: str) -> int | None:
        """Current training step: from the summary if present, else from history.

        wandb "shared" mode -- which prime-rl uses, and which is not optional
        for it -- does not populate `_step` in the run summary. Reading only the
        summary therefore reports "at step None: nothing has moved yet" for a
        run that has completed many steps, which makes every probe report
        `too_early` and leaves the monitor blind against precisely the stack it
        exists to watch.

        Observed live on 2026-09-08: a run with two completed steps, rich
        history (perf/throughput, effective_batch_size, errored_rollouts) and a
        summary containing only `_runtime`, `_timestamp` and `_wandb.runtime`.

        History carries both `step` (the TRAINING step) and `_step` (wandb's
        own counter, which increments once per log call). Prefer `step`.
        Reading `_step` here reported step 244 for a run on training step 2 and
        turned a ~277s/step rate into "17s/step" -- wrong by 16x, and wrong in
        the optimistic direction, which is the one that gets a 250-step run
        approved on a false budget.
        """

        run = self._run(run_path)
        try:
            rows = run.history(samples=200, pandas=False)
        except Exception as exc:  # noqa: BLE001 - any API shape counts as "unknown"
            raise WandbUnavailable(
                f"could not read history for {run_path}: {exc}"
            ) from exc

        for key in ("step", "_step"):
            best: int | None = None
            for row in rows or []:
                value = row.get(key)
                if isinstance(value, (int, float)) and value == value:  # not NaN
                    candidate = int(value)
                    if best is None or candidate > best:
                        best = candidate
            if best is not None:
                return best

        # Only now consider the summary, and only wandb's counter, since a run
        # that logged nothing to history has nothing better to offer.
        raw = self.summary(run_path).get("_step")
        return int(raw) if isinstance(raw, (int, float)) else None

    def step_rate_s(self, run_path: str) -> float | None:
        """Seconds per training step, measured BETWEEN logged steps.

        Not total_elapsed / step_count. That charges one-time startup to every
        step and is badly wrong exactly when someone is deciding whether a long
        run is affordable: on 2026-09-08 it reported 2032s/step for a run whose
        steps were taking 204-277s, because the elapsed time included a 45min
        provisioning lead and ~6min of vLLM load. Wrong by 8x, and the error
        shrinks only as the run gets longer -- so it is least accurate at the
        moment it is most used.

        Measuring first-to-last logged step removes the startup entirely.
        Returns None until two steps exist, because one step cannot define a
        rate.
        """
        run = self._run(run_path)
        try:
            rows = run.history(samples=500, pandas=False)
        except Exception as exc:  # noqa: BLE001
            raise WandbUnavailable(
                f"could not read history for {run_path}: {exc}"
            ) from exc

        # Shared mode means MULTIPLE writers (trainer and orchestrator) log
        # rows for the same run, interleaved and sometimes out of order. Two
        # rows a couple of seconds apart can straddle a step boundary, so raw
        # consecutive deltas produced "2s/step" for steps taking 204-277s.
        # Collapse to one timestamp per step first -- earliest wins, since that
        # is when the step was recorded rather than when the last writer
        # flushed.
        by_step: dict[int, float] = {}
        for row in rows or []:
            step = row.get("step")
            ts = row.get("_timestamp")
            if not isinstance(step, (int, float)) or not isinstance(ts, (int, float)):
                continue
            if step != step or ts != ts:  # NaN
                continue
            key = int(step)
            if key not in by_step or ts < by_step[key]:
                by_step[key] = float(ts)

        if len(by_step) < 2:
            return None
        ordered = sorted(by_step.items())

        # Median of per-step deltas. Not first-to-last: step 0 is logged at its
        # START, so a span measurement swallows step 0 plus warmup and read
        # 624s/step. Not elapsed/count either, which charges provisioning to
        # every step and read 2032s/step. The median additionally survives one
        # slow step (709s against 204s in the same run) without being dragged.
        deltas: list[float] = []
        for (s0, t0), (s1, t1) in zip(ordered, ordered[1:]):
            span_steps = s1 - s0
            span_s = t1 - t0
            if span_steps > 0 and span_s > 0:
                deltas.append(span_s / span_steps)
        if not deltas:
            return None
        deltas.sort()
        mid = len(deltas) // 2
        if len(deltas) % 2:
            return deltas[mid]
        return (deltas[mid - 1] + deltas[mid]) / 2.0

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
