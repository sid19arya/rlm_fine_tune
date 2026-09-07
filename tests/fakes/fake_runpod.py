"""In-memory fake of the RunPod REST API, served through an httpx MockTransport.

Reproduces the behaviours the failsafe layer depends on:

* `stop` pauses compute but the pod still exists and storage still bills;
  `terminate` deletes it. They are different, and the fake keeps them different.
* a **read-only key** answers every GET happily and returns 403 on writes. This
  is the fault that silently disarms every failsafe, so the suite has to be able
  to inject it.
* 5xx is transient (the client retries), 4xx is not.
* logs are Server-Sent Events with `id:` lines, so `Last-Event-ID` resume is
  testable.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

WRITE_METHODS = {"POST", "DELETE", "PATCH", "PUT"}


class FakePod:
    def __init__(
        self,
        pod_id: str,
        *,
        state: str = "RUNNING",
        gpu_type: str = "NVIDIA A40",
        gpu_count: int = 2,
        uptime_s: float = 0.0,
        cost_per_hr: float = 0.88,
    ) -> None:
        self.id = pod_id
        self.state = state
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        self.uptime_s = uptime_s
        self.cost_per_hr = cost_per_hr
        self.terminated = False
        self.logs: list[dict[str, str]] = []

    def emit(self, line: str, *, source: str = "stdout", ts: str | None = None) -> None:
        self.logs.append(
            {"source": source, "line": line, "ts": ts or f"{len(self.logs):06d}"}
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "desiredStatus": self.state,
            "machineType": self.gpu_type,
            "gpuCount": self.gpu_count,
            "uptimeSeconds": self.uptime_s,
            "costPerHr": self.cost_per_hr,
        }


class FakeRunPodServer:
    """Records every call so tests can assert *which* failsafe action fired."""

    def __init__(self, *, read_only_key: bool = False) -> None:
        self.pods: dict[str, FakePod] = {}
        self.read_only_key = read_only_key
        self.calls: list[tuple[str, str]] = []
        #: Number of 5xx responses to emit before succeeding, for retry tests.
        self.transient_failures = 0

    def add(self, pod: FakePod) -> FakePod:
        self.pods[pod.id] = pod
        return pod

    @property
    def stopped(self) -> list[str]:
        return [pod_id for method, pod_id in self.calls if method == "stop"]

    @property
    def terminated(self) -> list[str]:
        return [pod_id for method, pod_id in self.calls if method == "terminate"]

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def client(self, api_key: str = "test-key") -> httpx.Client:
        return httpx.Client(
            transport=self.transport(),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5.0,
        )

    # --- request handling ----------------------------------------------------

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method

        if self.transient_failures > 0:
            self.transient_failures -= 1
            return httpx.Response(503, text="upstream busy")

        if method in WRITE_METHODS and self.read_only_key:
            self.calls.append(("denied", path))
            return httpx.Response(403, json={"error": "insufficient scope: read-only key"})

        parts = [p for p in path.split("/") if p]

        # /v2/pods/{id}/logs  (SSE)
        if len(parts) == 4 and parts[0] == "v2" and parts[1] == "pods" and parts[3] == "logs":
            return self._logs(parts[2], request)

        if len(parts) >= 3 and parts[0] == "v1" and parts[1] == "pods":
            pod_id = parts[2]
            pod = self.pods.get(pod_id)
            if pod is None or pod.terminated:
                return httpx.Response(404, json={"error": "pod not found"})

            if len(parts) == 4 and parts[3] == "stop" and method == "POST":
                self.calls.append(("stop", pod_id))
                pod.state = "EXITED"
                return httpx.Response(200, json={"id": pod_id, "desiredStatus": "EXITED"})

            if len(parts) == 3 and method == "DELETE":
                self.calls.append(("terminate", pod_id))
                pod.terminated = True
                pod.state = "TERMINATED"
                return httpx.Response(200, json={"id": pod_id, "desiredStatus": "TERMINATED"})

            if len(parts) == 3 and method == "PATCH":
                self.calls.append(("write_probe", pod_id))
                return httpx.Response(200, json=pod.as_json())

            if len(parts) == 3 and method == "GET":
                self.calls.append(("status", pod_id))
                return httpx.Response(200, json=pod.as_json())

        return httpx.Response(404, json={"error": f"no route for {method} {path}"})

    def _logs(self, pod_id: str, request: httpx.Request) -> httpx.Response:
        pod = self.pods.get(pod_id)
        if pod is None:
            return httpx.Response(404, json={"error": "pod not found"})
        self.calls.append(("logs", pod_id))

        entries = pod.logs
        since = request.headers.get("Last-Event-ID")
        if since is not None:
            entries = [e for e in entries if e["ts"] > since]
        source = request.url.params.get("source")
        if source:
            entries = [e for e in entries if e["source"] == source]

        body = "".join(
            f"id: {e['ts']}\ndata: {json.dumps(e)}\n\n" for e in entries
        )
        return httpx.Response(
            200, text=body, headers={"Content-Type": "text/event-stream"}
        )
