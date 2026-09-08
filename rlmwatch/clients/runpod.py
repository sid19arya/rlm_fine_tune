"""RunPod REST client.

Scope is deliberately narrow: status, bounded logs, stop, terminate. This is the
only component that can spend or stop spending money, so it stays small enough
to audit in one sitting.

Two behaviours here are load-bearing:

* **403 is a config error, not a transient.** A read-only API key works fine for
  every read call and then fails at exactly the moment the failsafe layer tries
  to terminate a $30/hr pod. The startup gate calls `verify_write_scope()` so
  that discovery happens in the first 90 seconds instead.
* **Logs are bounded by default.** `tail_lines(n)` is the interface probes and
  agents use. The streaming iterator exists for diagnostics, and never feeds an
  LLM context directly.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx


def _age_seconds(stamp: object, *, now: datetime | None = None) -> float:
    """Seconds since a RunPod timestamp, or 0.0 if it cannot be parsed.

    RunPod returns e.g. ``2026-09-07 23:50:20.758 +0000 UTC`` -- a Go-style
    string that is not ISO 8601, so ``fromisoformat`` rejects it outright. The
    trailing zone name is dropped and the offset kept.

    Returning 0.0 on a parse failure is deliberate but dangerous, and callers
    should treat a 0 uptime on a pod that is plainly running as "unknown"
    rather than "free".
    """
    if not isinstance(stamp, str) or not stamp.strip():
        return 0.0
    text = stamp.strip()
    for suffix in (" UTC", "Z"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    # "+0000" -> "+00:00" so fromisoformat accepts the offset.
    match = re.search(r"([+-])(\d{2}):?(\d{2})$", text)
    if match:
        text = text[: match.start()].strip() + f"{match.group(1)}{match.group(2)}:{match.group(3)}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    return max(0.0, (reference - parsed).total_seconds())

DEFAULT_BASE_URL = "https://rest.runpod.io"

# Pod states RunPod reports. Anything not RUNNING while W&B says the run is
# Running is a disagreement between observers, which is itself a signal.
RUNNING = "RUNNING"


class RunPodError(RuntimeError):
    """Any non-retryable RunPod API failure."""


class RunPodAuthError(RunPodError):
    """401/403. Almost always a missing or read-only API key.

    Raised loudly and never retried: retrying a permissions failure just burns
    time while the pod keeps billing.
    """


@dataclass(frozen=True)
class PodStatus:
    id: str
    state: str
    gpu_type: str
    gpu_count: int
    uptime_s: float
    cost_per_hr: float
    raw: dict

    @property
    def is_running(self) -> bool:
        return self.state.upper() == RUNNING


@dataclass(frozen=True)
class LogLine:
    source: str  # "stdout" | "stderr" | "system"
    line: str
    ts: str  # also the SSE event id, used for Last-Event-ID resume


class RunPodClient:
    """Thin, retrying wrapper over the RunPod REST API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 4,
        client: httpx.Client | None = None,
        sleep=time.sleep,
    ) -> None:
        if not api_key:
            raise RunPodAuthError(
                "RUNPOD_API_KEY is empty. The monitor cannot stop a pod without it, "
                "so it refuses to start rather than pretending to protect the run."
            )
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._sleep = sleep
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    # --- plumbing ------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Issue a request, retrying 5xx and transport errors with jittered backoff.

        4xx is never retried: it will not become correct on the second attempt.
        """
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                last_exc = exc
            else:
                if response.status_code in (401, 403):
                    raise RunPodAuthError(
                        f"{method} {path} returned {response.status_code}. The API key is "
                        f"missing, expired, or read-only. A read-only key can poll but "
                        f"cannot stop or terminate, which disables every failsafe in this "
                        f"library. Fix the key scope before running anything expensive."
                    )
                if response.status_code < 500:
                    return response
                last_exc = RunPodError(
                    f"{method} {path} -> {response.status_code}: {response.text[:200]}"
                )

            if attempt < self.max_retries - 1:
                backoff = min(2.0**attempt, 8.0) * (0.5 + random.random())
                self._sleep(backoff)

        raise RunPodError(f"{method} {path} failed after {self.max_retries} attempts: {last_exc}")

    def _json(self, method: str, path: str, **kwargs: Any) -> dict:
        response = self._request(method, path, **kwargs)
        if response.status_code == 404:
            raise RunPodError(f"{method} {path} -> 404: pod not found")
        if response.status_code >= 400:
            raise RunPodError(f"{method} {path} -> {response.status_code}: {response.text[:200]}")
        if not response.content:
            return {}
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise RunPodError(
                f"{method} {path}: response was not JSON: {response.text[:200]}"
            ) from exc

    # --- reads ---------------------------------------------------------------

    def status(self, pod_id: str) -> PodStatus:
        """Pod state, GPU type/count and uptime."""
        data = self._json("GET", f"/v1/pods/{pod_id}")
        machine = data.get("machine") or {}
        uptime = data.get("uptimeSeconds")
        if uptime is None:
            uptime = (data.get("runtime") or {}).get("uptimeInSeconds") or None
        if not uptime:
            # Neither field is populated on a live pod -- observed on a running
            # 2x A40: `uptimeSeconds` absent and `runtime` null. Falling back to
            # 0 would be catastrophic rather than merely wrong: every cost probe
            # computes spend as uptime x rate, so a permanent 0 means spend is
            # always $0.00 and the budget cap can never fire. The one safety
            # mechanism that has to work without a human present would be
            # silently inert.
            #
            # `createdAt` is the correct clock anyway: billing starts when the
            # pod is provisioned, not when it finishes booting.
            uptime = _age_seconds(data.get("createdAt") or data.get("lastStartedAt"))
        gpu_type = (
            data.get("machineType") or machine.get("gpuTypeId") or data.get("gpuTypeId") or ""
        )
        return PodStatus(
            id=data.get("id", pod_id),
            state=str(data.get("desiredStatus") or data.get("status") or "UNKNOWN").upper(),
            gpu_type=str(gpu_type),
            gpu_count=int(data.get("gpuCount") or 0),
            uptime_s=float(uptime),
            cost_per_hr=float(data.get("costPerHr") or 0.0),
            raw=data,
        )

    def stream_logs(
        self,
        pod_id: str,
        *,
        since: str | None = None,
        source: str | None = None,
        limit: int | None = None,
    ) -> Iterator[LogLine]:
        """Iterate the pod's Server-Sent Events log stream.

        Event ids are timestamps, so `since` resumes via `Last-Event-ID`.

        `limit` exists because unbounded log streaming into an agent context is
        an explicit non-goal of this library. Prefer `tail_lines`.
        """
        headers = {"Accept": "text/event-stream"}
        if since:
            headers["Last-Event-ID"] = since
        params = {"source": source} if source else None

        emitted = 0
        with self._client.stream(
            "GET", f"{self.base_url}/v2/pods/{pod_id}/logs", headers=headers, params=params
        ) as response:
            if response.status_code in (401, 403):
                raise RunPodAuthError(
                    f"log stream returned {response.status_code}: check key scope"
                )
            if response.status_code >= 400:
                raise RunPodError(f"log stream returned {response.status_code}")

            event_id: str | None = None
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                if raw_line.startswith("id:"):
                    event_id = raw_line[3:].strip()
                    continue
                if not raw_line.startswith("data:"):
                    continue
                payload = raw_line[5:].strip()
                if not payload:
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    obj = {"source": "unknown", "line": payload, "ts": event_id or ""}
                yield LogLine(
                    source=str(obj.get("source", "unknown")),
                    line=str(obj.get("line", "")),
                    ts=str(obj.get("ts") or event_id or ""),
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    def tail_lines(self, pod_id: str, n: int = 500, *, source: str | None = None) -> list[LogLine]:
        """The last `n` log lines, as a bounded list.

        This is the interface diagnostics and probes use. `n` defaults to the
        500 lines the snapshot bundle wants and is hard-capped so that a caller
        cannot accidentally pull a gigabyte of logs into memory or a prompt.
        """
        n = max(1, min(int(n), 5000))
        buffer: list[LogLine] = []
        for entry in self.stream_logs(pod_id, source=source):
            buffer.append(entry)
            if len(buffer) > n:
                buffer.pop(0)
        return buffer

    # --- writes --------------------------------------------------------------

    def verify_write_scope(self, pod_id: str) -> bool:
        """Prove the key can write, by writing.

        Assuming write scope and finding out otherwise at kill time defeats the
        entire failsafe layer, so the startup gate calls this before any GPU
        time is spent. The write is benign -- a no-op tag/label update on the
        pod itself -- and any auth failure propagates as RunPodAuthError.
        """
        self._json("PATCH", f"/v1/pods/{pod_id}", json={"name": None})
        return True

    def stop(self, pod_id: str) -> dict:
        """Pause compute. **Storage keeps billing.** Not the same as terminate."""
        return self._json("POST", f"/v1/pods/{pod_id}/stop")

    def terminate(self, pod_id: str) -> dict:
        """Delete the pod and every disk on it that is not a network volume.

        This is the only call that actually stops the meter. A clean process
        exit does not: the container restarts and billing continues.
        """
        return self._json("DELETE", f"/v1/pods/{pod_id}")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RunPodClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
