"""Client behaviour, against the fakes. No network, no cloud resources."""

from __future__ import annotations

import pytest

from rlmwatch.clients.runpod import RunPodAuthError, RunPodClient, RunPodError
from rlmwatch.clients.wandb import WandbClient, WandbUnavailable
from tests.fakes.fake_runpod import FakePod, FakeRunPodServer
from tests.fakes.fake_wandb import FakeRun, FakeWandbApi, gpu_system_metrics


@pytest.fixture
def server():
    srv = FakeRunPodServer()
    srv.add(FakePod("pod-1", uptime_s=3600.0))
    return srv


@pytest.fixture
def runpod(server):
    return RunPodClient("test-key", client=server.client(), sleep=lambda _: None)


class TestRunPodClient:
    def test_status_reports_state_gpus_and_uptime(self, runpod):
        status = runpod.status("pod-1")
        assert status.is_running
        assert status.gpu_count == 2
        assert status.gpu_type == "NVIDIA A40"
        assert status.uptime_s == 3600.0

    def test_stop_and_terminate_are_different_operations(self, server, runpod):
        runpod.stop("pod-1")
        assert server.stopped == ["pod-1"]
        assert server.pods["pod-1"].terminated is False  # storage still billing

        runpod.terminate("pod-1")
        assert server.terminated == ["pod-1"]
        assert server.pods["pod-1"].terminated is True

    def test_empty_api_key_refuses_to_construct(self):
        with pytest.raises(RunPodAuthError):
            RunPodClient("")

    def test_read_only_key_fails_loudly_on_write_scope_check(self):
        """The fault that silently disarms every failsafe."""
        srv = FakeRunPodServer(read_only_key=True)
        srv.add(FakePod("pod-1"))
        client = RunPodClient("ro-key", client=srv.client(), sleep=lambda _: None)

        assert client.status("pod-1").is_running  # reads work fine, which is the trap

        with pytest.raises(RunPodAuthError, match="read-only"):
            client.verify_write_scope("pod-1")

    def test_read_only_key_is_never_retried(self):
        srv = FakeRunPodServer(read_only_key=True)
        srv.add(FakePod("pod-1"))
        calls = []
        client = RunPodClient("ro-key", client=srv.client(), sleep=calls.append)
        with pytest.raises(RunPodAuthError):
            client.terminate("pod-1")
        assert calls == [], "a permissions failure must not be retried with backoff"

    def test_write_scope_check_passes_with_a_write_capable_key(self, server, runpod):
        assert runpod.verify_write_scope("pod-1") is True

    def test_5xx_is_retried_then_succeeds(self, server, runpod):
        server.transient_failures = 2
        assert runpod.status("pod-1").is_running

    def test_persistent_5xx_eventually_raises(self, server, runpod):
        server.transient_failures = 99
        with pytest.raises(RunPodError, match="failed after"):
            runpod.status("pod-1")

    def test_missing_pod_is_a_404_not_a_retry_loop(self, runpod):
        with pytest.raises(RunPodError, match="404"):
            runpod.status("pod-nonexistent")


class TestRunPodLogs:
    def test_tail_lines_is_bounded(self, server, runpod):
        pod = server.pods["pod-1"]
        for i in range(1200):
            pod.emit(f"line {i}")

        tail = runpod.tail_lines("pod-1", n=500)
        assert len(tail) == 500
        assert tail[-1].line == "line 1199", "tail must be the newest lines, not the oldest"

    def test_tail_is_hard_capped_even_if_a_caller_asks_for_more(self, server, runpod):
        pod = server.pods["pod-1"]
        for i in range(20):
            pod.emit(f"line {i}")
        assert len(runpod.tail_lines("pod-1", n=10_000_000)) == 20

    def test_stream_resumes_from_last_event_id(self, server, runpod):
        pod = server.pods["pod-1"]
        for i in range(10):
            pod.emit(f"line {i}")

        first = list(runpod.stream_logs("pod-1", limit=4))
        assert [entry.line for entry in first] == ["line 0", "line 1", "line 2", "line 3"]

        resumed = list(runpod.stream_logs("pod-1", since=first[-1].ts))
        assert [entry.line for entry in resumed] == [f"line {i}" for i in range(4, 10)]

    def test_stream_can_filter_by_source(self, server, runpod):
        pod = server.pods["pod-1"]
        pod.emit("normal", source="stdout")
        pod.emit("CUDA error", source="stderr")
        errors = list(runpod.stream_logs("pod-1", source="stderr"))
        assert [entry.line for entry in errors] == ["CUDA error"]


class TestWandbClient:
    def make(self, *, state="running", summary=None, now=1_000_000.0):
        api = FakeWandbApi()
        api.add(FakeRun("e/p/r", state=state, summary=summary or {}))
        return api, WandbClient(api=api, now=lambda: now)

    def test_state_is_lowercased(self):
        _, client = self.make(state="Crashed")
        assert client.state("e/p/r") == "crashed"

    def test_last_progress_age_measures_the_training_loop_not_the_heartbeat(self):
        """The deadlocked-loop case: state Running, progress stale."""
        api, client = self.make(state="running", summary={"_timestamp": 1_000_000.0 - 4200})
        assert client.state("e/p/r") == "running"
        assert client.last_progress_age("e/p/r") == pytest.approx(4200.0)

    def test_missing_timestamp_is_unavailable_not_zero_age(self):
        _, client = self.make(summary={})
        with pytest.raises(WandbUnavailable):
            client.last_progress_age("e/p/r")

    def test_network_outage_raises_unavailable_rather_than_lying(self):
        api, client = self.make()
        api.unavailable = True
        client.invalidate()
        with pytest.raises(WandbUnavailable):
            client.state("e/p/r")

    def test_run_object_is_cached_to_keep_sentinel_cost_down(self):
        api, client = self.make(summary={"_timestamp": 1_000_000.0})
        for _ in range(10):
            client.state("e/p/r")
        assert api.call_count == 1

    def test_invalidate_forces_a_fresh_read_for_confirmation_polls(self):
        api, client = self.make(summary={"_timestamp": 1_000_000.0})
        client.state("e/p/r")
        client.invalidate("e/p/r")
        client.state("e/p/r")
        assert api.call_count == 2

    def test_metric_window_drops_non_finite_samples(self):
        api = FakeWandbApi()
        run = FakeRun("e/p/r")
        for value in [0.4, float("nan"), 0.5, float("inf"), 0.6]:
            run.log({"reward/std": value})
        api.add(run)
        client = WandbClient(api=api)
        assert client.metric_window("e/p/r", "reward/std", n=10) == [0.4, 0.5, 0.6]

    def test_raw_metric_window_keeps_non_finite_for_the_loss_probe(self):
        api = FakeWandbApi()
        run = FakeRun("e/p/r")
        run.log({"train/loss": 1.0})
        run.log({"train/loss": float("nan")})
        api.add(run)
        client = WandbClient(api=api)
        values = client.raw_metric_window("e/p/r", "train/loss", n=10)
        assert len(values) == 2 and values[1] != values[1]  # NaN preserved

    def test_system_metrics_are_parsed_per_device(self):
        _, client = self.make(summary=gpu_system_metrics(3.0, gpus=2))
        metrics = client.system_metrics("e/p/r")
        assert metrics.gpu_util_pct == {0: 3.0, 1: 3.0}
        assert metrics.mean_util() == 3.0
        assert metrics.min_free_mem_pct() == pytest.approx(30.0)

    def test_confirm_requires_the_predicate_to_hold_twice(self):
        """A network blip must not be able to terminate a healthy pod."""
        api, client = self.make(summary={"_timestamp": 1_000_000.0})
        readings = iter([True, False])
        assert client.confirm("e/p/r", lambda: next(readings), delay_s=0,
                              sleep=lambda _: None) is False

    def test_confirm_passes_when_the_condition_persists(self):
        api, client = self.make(summary={"_timestamp": 1_000_000.0})
        assert client.confirm("e/p/r", lambda: True, delay_s=0, sleep=lambda _: None) is True
