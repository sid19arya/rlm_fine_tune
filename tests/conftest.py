"""Shared fixtures. Everything runs against fakes -- no GPU, no network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from rlmwatch.clients.runpod import RunPodClient
from rlmwatch.clients.wandb import WandbClient
from rlmwatch.config import from_dict
from rlmwatch.probes.base import Context
from tests.fakes.fake_runpod import FakePod, FakeRunPodServer
from tests.fakes.fake_wandb import FakeRun, FakeWandbApi

NOW = 1_700_000_000.0
RUN_PATH = "acme/rlm/run-001"
POD_ID = "pod-1"

BASE_CONFIG: dict[str, Any] = {
    "run": {"name": "rlm-ft-001", "regime": "rl", "wandb": RUN_PATH, "pod_id": POD_ID},
    "expect": {"gpu_type": "NVIDIA A40", "gpu_count": 2, "min_free_vram_gb": 40,
               "min_disk_gb": 50, "checkpoint_dir": "/workspace/ckpt"},
    "budget": {"hourly_rate_usd": 0.88, "max_usd": 5.0, "warn_pct": 75,
               "max_wall_clock_h": 3.0},
    "failsafe": {"on_terminal": "stop", "confirm_delay_s": 1, "dead_mans_timeout_s": 1800},
    "stall_threshold": {"default": 300, "rollout": "auto", "checkpoint": 900},
}


@pytest.fixture
def config_dict() -> dict[str, Any]:
    """A deep-ish copy callers can mutate before building a config."""
    return {k: dict(v) if isinstance(v, dict) else v for k, v in BASE_CONFIG.items()}


@pytest.fixture
def cfg(config_dict):
    return from_dict(config_dict)


@pytest.fixture
def wandb_api():
    api = FakeWandbApi()
    api.add(FakeRun(RUN_PATH, state="running", summary={"_timestamp": NOW}))
    return api


@pytest.fixture
def wandb_run(wandb_api) -> FakeRun:
    return wandb_api.runs[RUN_PATH]


@pytest.fixture
def wandb_client(wandb_api) -> WandbClient:
    # cache_ttl_s=0 so tests see their own mutations immediately.
    return WandbClient(api=wandb_api, now=lambda: NOW, cache_ttl_s=0.0)


@pytest.fixture
def runpod_server() -> FakeRunPodServer:
    server = FakeRunPodServer()
    server.add(FakePod(POD_ID, uptime_s=3600.0, cost_per_hr=0.88))
    return server


@pytest.fixture
def runpod_client(runpod_server) -> RunPodClient:
    return RunPodClient("k", client=runpod_server.client(), sleep=lambda _: None)


@pytest.fixture
def make_ctx(cfg, wandb_client, runpod_client):
    """Build a probe Context, overriding any field per test."""

    def _make(**overrides: Any) -> Context:
        started_at = overrides.pop(
            "started_at", datetime.now(timezone.utc) - timedelta(hours=1)
        )
        ctx = Context(
            cfg=overrides.pop("cfg", cfg),
            wandb=overrides.pop("wandb", wandb_client),
            runpod=overrides.pop("runpod", runpod_client),
            phase=overrides.pop("phase", None),
            started_at=started_at,
            baselines=overrides.pop("baselines", {}),
            local=overrides.pop("local", {}),
        )
        assert not overrides, f"unexpected Context overrides: {sorted(overrides)}"
        return ctx

    return _make


def log_series(run: FakeRun, key: str, values: list[float]) -> None:
    """Append a metric series to a fake run's history."""
    for value in values:
        run.log({key: value}, timestamp=NOW)
