"""Section 8 acceptance criteria.

The spec is explicit: *the library is not done when it is written, it is done
when it survives injected faults.* Each test below is one row of that table.

One test class per fault, named for the fault. The final class covers the
additional bar -- no termination on a single unconfirmed reading, sentinel cost,
and zero false terminations across a 24-hour soak on a healthy run.
"""

from __future__ import annotations

import pytest

from rlmwatch.actions import Level
from rlmwatch.clients.runpod import RunPodAuthError, RunPodClient
from rlmwatch.notify import Notifier, RecordingSink
from rlmwatch.probes.rlm import RewardStdProbe
from rlmwatch.probes.startup import (
    DiskSpaceProbe,
    RunPodWriteScopeProbe,
    StartupContext,
    StartupHooks,
    gate,
)
from tests.chaos.harness import POD_ID, build_run
from tests.fakes.fake_runpod import FakePod, FakeRunPodServer

pytestmark = pytest.mark.chaos


class TestKill9Trainer:
    """kill -9 the trainer -> detected < 3 min, pod terminated. Owner: sentinel."""

    def test_detected_and_terminated_within_three_minutes(self):
        chaos = build_run()
        for _ in range(5):
            chaos.log_step()
            chaos.clock.advance(30)

        chaos.kill_9_trainer()
        killed_at = chaos.clock()

        sentinel = chaos.sentinel()
        detection_s = None
        while chaos.clock() - killed_at < 600:
            sentinel.tick()
            if chaos.terminated:
                # Measured at the moment of action, not after the next poll
                # sleep: the interval the sentinel would have waited had it
                # done nothing is not part of the detection latency.
                detection_s = chaos.clock() - killed_at
                break
            chaos.clock.advance(chaos.cfg.poll_interval_s)

        assert chaos.terminated, "a SIGKILLed trainer must not leave a billing pod"
        assert detection_s < 180, f"detected only after {detection_s:.0f}s"
        # Almost all of that budget is the deliberate confirmation wait, which
        # is the price of never killing a healthy pod on one bad reading.
        assert detection_s >= chaos.cfg.failsafe.confirm_delay_s

    def test_the_disagreement_is_what_catches_it(self):
        """Pod RUNNING + W&B crashed. Neither observer sees this alone."""
        chaos = build_run()
        chaos.log_step()
        chaos.kill_9_trainer()

        chaos.sentinel().tick()
        probes = chaos.alert_probes()
        assert "liveness.observer_agreement" in probes
        assert chaos.runpod_server.pods[POD_ID].state != "RUNNING"

    def test_no_handler_ran_so_nothing_else_would_have_stopped_it(self):
        """SIGKILL runs no finally block: without the sentinel it bills forever."""
        chaos = build_run()
        chaos.log_step()
        chaos.kill_9_trainer()
        assert chaos.trainer.checkpoint_requests == 0
        assert not chaos.terminated, "nothing in-pod could have reacted"


class TestSigstopTrainer:
    """SIGSTOP the trainer -> detected within the phase threshold, py-spy dump
    captured, pod terminated. Owner: sentinel."""

    def test_a_hang_that_looks_perfectly_healthy_is_still_caught(self):
        chaos = build_run()
        for _ in range(3):
            chaos.log_step()
            chaos.clock.advance(60)

        chaos.sigstop_trainer()
        # Everything a naive monitor would check still reads healthy.
        assert chaos.run.state == "running"
        assert chaos.runpod_server.pods[POD_ID].state == "RUNNING"

        sentinel = chaos.sentinel()
        hung_at = chaos.clock()
        while chaos.clock() - hung_at < 3600 and not chaos.terminated:
            sentinel.tick()
            chaos.clock.advance(chaos.cfg.poll_interval_s)

        assert chaos.terminated
        assert "liveness.progress_age" in chaos.alert_probes()

    def test_a_py_spy_dump_is_captured_before_anything_is_killed(self):
        """Killed without a dump teaches you nothing and it will hang again."""
        chaos = build_run({"failsafe": {"confirm_delay_s": 1}})
        chaos.log_step()
        chaos.sigstop_trainer()
        chaos.clock.advance(4000)

        sentinel = chaos.sentinel()
        for _ in range(3):
            sentinel.tick()
            if chaos.terminated:
                break
            chaos.clock.advance(60)

        assert chaos.terminated
        snapshots = chaos.snapshots()
        assert snapshots, "no diagnostic bundle was collected"
        stacks = "".join(next(iter(s.stacks.values())) for s in snapshots if s.stacks)
        assert "_wait_for_rollout" in stacks, "the stack trace is the whole point"
        assert any(c[0] == "py-spy" and "--nonblocking" in c for c in chaos.commands_run)

    def test_the_hang_is_measured_against_the_phase_threshold(self):
        """A rollout gets 4x its baseline; an update does not."""
        chaos = build_run()
        chaos.log_step()
        watchdog = chaos.watchdog(baselines={"rollout": 200.0})
        watchdog.phase("update")
        assert watchdog.context.stall_threshold_s() == 300
        watchdog.phase("rollout")
        assert watchdog.context.stall_threshold_s() == 800


class TestWandbNetworkCut:
    """Network cut from pod to W&B -> unknown, not fail; no termination inside
    unknown_tolerance; recovers cleanly. Owner: both."""

    def test_an_outage_produces_unknown_not_fail(self):
        chaos = build_run()
        chaos.log_step()
        chaos.cut_wandb_network()

        result = chaos.sentinel().tick()
        wandb_verdicts = [v for v in result.verdicts
                          if v.probe.startswith(("liveness.progress_age", "rlm.", "health."))]
        assert wandb_verdicts
        assert all(v.status != "fail" for v in wandb_verdicts), \
            "a network blip must never be reported as a run failure"

    def test_nothing_is_terminated_inside_the_unknown_tolerance(self):
        chaos = build_run()
        chaos.log_step()
        chaos.cut_wandb_network()

        sentinel = chaos.sentinel()
        deadline = chaos.clock() + chaos.cfg.failsafe.unknown_tolerance_s - 60
        while chaos.clock() < deadline:
            sentinel.tick()
            chaos.clock.advance(chaos.cfg.poll_interval_s)

        assert not chaos.terminated, "a healthy pod was killed over a W&B outage"

    def test_the_run_recovers_cleanly_when_the_network_comes_back(self):
        chaos = build_run()
        chaos.log_step()
        chaos.cut_wandb_network()

        sentinel = chaos.sentinel()
        for _ in range(5):
            sentinel.tick()
            chaos.clock.advance(60)

        chaos.restore_wandb_network()
        chaos.log_step()
        result = sentinel.tick()

        assert not chaos.terminated
        progress = next(v for v in result.verdicts if v.probe == "liveness.progress_age")
        assert progress.status == "ok"

    def test_sustained_blindness_is_reported_as_its_own_incident(self):
        """A monitor that cannot see is an incident -- but never a kill reason."""
        chaos = build_run()
        chaos.log_step()
        chaos.cut_wandb_network()

        sentinel = chaos.sentinel()
        for _ in range(40):  # well past unknown_tolerance_s
            sentinel.tick()
            chaos.clock.advance(60)

        promoted = [a for a in chaos.alerts
                    if a.verdict.evidence.get("promoted_from") == "unknown"]
        assert promoted, "sustained unknown was never surfaced"
        assert all(a.verdict.status == "warn" for a in promoted)


class TestDiskFull:
    """Fill the disk -> startup gate fails, or a mid-run warn before the
    checkpoint write fails. Owner: watchdog."""

    def test_the_startup_gate_refuses_to_start_on_a_full_disk(self):
        chaos = build_run()

        class FullDisk:
            def gpus(self):
                from rlmwatch.probes.startup import GpuInfo

                return [GpuInfo(i, "NVIDIA A40", 47.0, 48.0) for i in range(2)]

            def torch_device_count(self):
                return 2

            def xid_errors(self):
                return []

            def nccl_all_reduce(self, timeout_s):
                return True

            def free_disk_gb(self, path, quota_gb=None):
                return 0.4

        from rlmwatch.probes.base import Context

        sctx = StartupContext(
            ctx=Context(cfg=chaos.cfg, wandb=chaos.wandb, runpod=chaos.runpod),
            hardware=FullDisk(), hooks=StartupHooks(),
            notifier=Notifier(chaos.cfg, sinks=[RecordingSink()]),
        )
        verdict = DiskSpaceProbe().inspect(sctx)
        assert verdict.status == "fail"
        assert verdict.evidence["free_gb"] == 0.4

    def test_a_full_disk_fails_the_whole_gate_before_gpu_time_is_spent(self):
        chaos = build_run()

        class FullDisk:
            def gpus(self):
                from rlmwatch.probes.startup import GpuInfo

                return [GpuInfo(i, "NVIDIA A40", 47.0, 48.0) for i in range(2)]

            def torch_device_count(self):
                return 2

            def xid_errors(self):
                return []

            def nccl_all_reduce(self, timeout_s):
                return True

            def free_disk_gb(self, path, quota_gb=None):
                return 0.0

        from rlmwatch.probes.base import Context

        sctx = StartupContext(
            ctx=Context(cfg=chaos.cfg, wandb=chaos.wandb, runpod=chaos.runpod),
            hardware=FullDisk(), hooks=StartupHooks(),
            notifier=Notifier(chaos.cfg, sinks=[RecordingSink()]),
        )
        result = gate(sctx)
        assert not result.passed
        assert result.failure.probe == "startup.disk_space"
        assert "startup.train_step" not in [v.probe for v in result.verdicts]


class TestRewardHeldConstant:
    """Reward held constant -> the reward_std probe fails within
    reward_std_patience steps. Owner: watchdog."""

    def test_zero_variance_fails_within_patience(self):
        chaos = build_run({"health": {"reward_std_patience": 20}})
        for _ in range(20):
            chaos.log_step(**{"reward/std": 0.0})
            chaos.clock.advance(30)

        verdict = RewardStdProbe().check(chaos.watchdog().context)
        assert verdict.status == "fail"
        assert verdict.evidence["collapsed_of_window"] == 20

    def test_it_does_not_fire_before_patience_is_exhausted(self):
        """One degenerate batch is normal; the probe must not be trigger-happy."""
        chaos = build_run({"health": {"reward_std_patience": 20}})
        for _ in range(15):
            chaos.log_step()
        for _ in range(4):
            chaos.log_step(**{"reward/std": 0.0})

        assert RewardStdProbe().check(chaos.watchdog().context).status != "fail"

    def test_the_run_is_halted_and_the_checkpoint_saved(self):
        """Training is a no-op; save what exists rather than burning more spend."""
        chaos = build_run({"health": {"reward_std_patience": 5},
                           "failsafe": {"confirm_delay_s": 1}})
        for _ in range(6):
            chaos.log_step(**{"reward/std": 0.0})
            chaos.clock.advance(30)

        watchdog = chaos.watchdog(probes=(RewardStdProbe(),))
        watchdog.tick()

        assert chaos.ladder.halt_requested is True
        assert chaos.trainer.checkpoints_saved >= 1

    def test_every_dashboard_still_looks_normal(self):
        """The reason this probe exists: nothing else notices."""
        chaos = build_run({"health": {"reward_std_patience": 5}})
        for _ in range(6):
            chaos.log_step(**{"reward/std": 0.0})
            chaos.clock.advance(30)

        result = chaos.sentinel().tick()
        by_probe = {v.probe: v.status for v in result.verdicts}
        assert by_probe["liveness.progress_age"] == "ok"
        assert by_probe["liveness.gpu_util"] == "ok"
        assert by_probe["liveness.observer_agreement"] == "ok"
        assert by_probe["rlm.reward_std"] == "fail"


class TestRolloutTenTimesBaseline:
    """Rollout duration 10x baseline -> stall detected, without false-alarming on
    normal rollouts. Owner: both."""

    def test_a_ten_times_rollout_fails(self):
        chaos = build_run()
        for _ in range(10):
            chaos.log_step(**{"rollout/duration_s": 2000.0})

        watchdog = chaos.watchdog(baselines={"rollout": 200.0})
        from rlmwatch.probes.rlm import RolloutDurationProbe

        verdict = RolloutDurationProbe().check(watchdog.context)
        assert verdict.status == "fail"
        assert verdict.evidence["ratio"] == 10.0

    def test_a_normal_rollout_does_not_false_alarm(self):
        """The other half of the requirement, and the harder one."""
        chaos = build_run()
        for duration in (180.0, 210.0, 195.0, 240.0, 205.0):
            chaos.log_step(**{"rollout/duration_s": duration})

        watchdog = chaos.watchdog(baselines={"rollout": 200.0})
        from rlmwatch.probes.rlm import RolloutDurationProbe

        assert RolloutDurationProbe().check(watchdog.context).status == "ok"

    def test_a_long_but_healthy_rollout_phase_is_not_a_stall(self):
        chaos = build_run()
        chaos.log_step()
        chaos.clock.advance(700)  # longer than the 300s default, inside 4x200s

        watchdog = chaos.watchdog(baselines={"rollout": 200.0})
        watchdog.phase("rollout")
        from rlmwatch.probes.liveness import ProgressAgeProbe

        assert ProgressAgeProbe().check(watchdog.context).status != "fail"


class TestNormalCompletion:
    """Training completes normally -> Finished, checkpoint saved, pod terminated,
    spend reported. Owner: watchdog."""

    def test_a_clean_exit_still_terminates_the_pod(self):
        """The container restarts when the entrypoint ends; the meter runs on."""
        chaos = build_run()
        with chaos.watchdog(probes=()):
            for _ in range(20):
                chaos.log_step()
                chaos.clock.advance(30)
            # The trainer's own final save, as it finishes. This is the
            # trainer's responsibility, not the monitor's.
            chaos.trainer.request_checkpoint()
            chaos.trainer.wait_for_checkpoint(60)
        chaos.finish_normally()

        assert chaos.trainer.checkpoints_saved >= 1
        assert chaos.terminated
        assert chaos.runpod_server.stopped == [POD_ID]

    def test_shutdown_requests_a_checkpoint_without_blocking_on_it(self):
        """At shutdown the loop is already over, so there is no next safe point.

        Waiting checkpoint_timeout_s for one would stall every clean exit for
        ten minutes of billing and then terminate anyway.
        """
        chaos = build_run({"failsafe": {"checkpoint_timeout_s": 600}})
        chaos.trainer.can_checkpoint = False  # would block if shutdown waited
        started = chaos.clock()

        with chaos.watchdog(probes=()):
            chaos.log_step()

        assert chaos.trainer.checkpoint_requests >= 1, "the monitor should still ask"
        assert chaos.clock() - started < 600, "shutdown must not wait for a safe point"
        assert chaos.terminated

    def test_completion_reports_spend(self):
        chaos = build_run()
        for _ in range(20):
            chaos.log_step()
            chaos.clock.advance(180)

        result = chaos.sentinel().tick()
        assert result.spend_usd is not None and result.spend_usd > 0
        assert result.projected_usd is not None

    def test_a_finished_run_with_a_live_pod_is_itself_an_alert(self):
        chaos = build_run()
        chaos.log_step()
        chaos.finish_normally()

        result = chaos.sentinel().tick()
        agreement = next(v for v in result.verdicts
                         if v.probe == "liveness.observer_agreement")
        assert agreement.status == "fail"
        assert "does not stop billing" in agreement.detail


class TestSentinelKilled:
    """Sentinel process killed -> the external heartbeat service pages within its
    own window. Owner: external."""

    def test_a_healthy_tick_pings_the_external_heartbeat(self):
        chaos = build_run({"notify": {"heartbeat_url": "https://hc-ping.test/abc"}})
        chaos.log_step()

        pings: list[tuple[str, bytes]] = []

        class FakeResponse:
            status_code = 200

        class FakeHttp:
            def post(self, url, content=None, json=None):
                pings.append((url, content))
                return FakeResponse()

        chaos.ladder.notifier._client = FakeHttp()
        chaos.sentinel().tick()

        assert pings, "no heartbeat was sent; the sentinel's silence would go unnoticed"
        assert pings[0][0].startswith("https://hc-ping.test/abc")

    def test_a_dead_sentinel_stops_pinging_which_is_the_signal(self):
        """This library deliberately does not watch its own watcher.

        A fourth layer watching the third is the wrong answer; the sentinel's
        silence is delegated to an external dead-man's-switch service.
        """
        chaos = build_run({"notify": {"heartbeat_url": "https://hc-ping.test/abc"}})
        pings: list[str] = []

        class FakeResponse:
            status_code = 200

        class FakeHttp:
            def post(self, url, content=None, json=None):
                pings.append(url)
                return FakeResponse()

        chaos.ladder.notifier._client = FakeHttp()
        sentinel = chaos.sentinel()
        chaos.log_step()
        sentinel.tick()
        before = len(pings)

        # Sentinel process dies: no more ticks, so no more pings.
        chaos.clock.advance(3600)
        assert len(pings) == before

    def test_a_failing_tick_pings_the_fail_endpoint(self):
        chaos = build_run({"notify": {"heartbeat_url": "https://hc-ping.test/abc"}})
        chaos.log_step()
        chaos.kill_9_trainer()

        pings: list[str] = []

        class FakeResponse:
            status_code = 200

        class FakeHttp:
            def post(self, url, content=None, json=None):
                pings.append(url)
                return FakeResponse()

        chaos.ladder.notifier._client = FakeHttp()
        chaos.sentinel().tick()
        assert any(url.endswith("/fail") for url in pings)


class TestReadOnlyApiKey:
    """Read-only API key -> the startup gate fails loudly, before any GPU time is
    spent. Owner: startup."""

    def test_the_gate_fails_before_the_model_is_ever_loaded(self):
        chaos = build_run()
        server = FakeRunPodServer(read_only_key=True)
        server.add(FakePod(POD_ID))
        readonly = RunPodClient("ro", client=server.client(), sleep=lambda _: None)

        from rlmwatch.probes.base import Context

        loaded = []
        sctx = StartupContext(
            ctx=Context(cfg=chaos.cfg, wandb=chaos.wandb, runpod=readonly),
            hardware=_HealthyHardware(), notifier=Notifier(chaos.cfg, sinks=[RecordingSink()]),
            hooks=StartupHooks(load_model=lambda: loaded.append(1) or {"param_count": 1}),
        )
        result = gate(sctx)
        assert not result.passed
        assert result.failure.probe == "startup.runpod_write_scope"
        assert loaded == [], "the gate must fail before any GPU time is spent"

    def test_the_failure_names_the_consequence(self):
        chaos = build_run()
        server = FakeRunPodServer(read_only_key=True)
        server.add(FakePod(POD_ID))
        readonly = RunPodClient("ro", client=server.client(), sleep=lambda _: None)

        from rlmwatch.probes.base import Context

        sctx = StartupContext(
            ctx=Context(cfg=chaos.cfg, wandb=chaos.wandb, runpod=readonly),
            hardware=_HealthyHardware(), hooks=StartupHooks(),
        )
        verdict = RunPodWriteScopeProbe().inspect(sctx)
        assert "kill time" in verdict.detail

    def test_reads_keep_working_which_is_exactly_the_trap(self):
        server = FakeRunPodServer(read_only_key=True)
        server.add(FakePod(POD_ID))
        readonly = RunPodClient("ro", client=server.client(), sleep=lambda _: None)
        assert readonly.status(POD_ID).is_running
        with pytest.raises(RunPodAuthError):
            readonly.terminate(POD_ID)


class _HealthyHardware:
    def gpus(self):
        from rlmwatch.probes.startup import GpuInfo

        return [GpuInfo(i, "NVIDIA A40", 47.0, 48.0) for i in range(2)]

    def torch_device_count(self):
        return 2

    def xid_errors(self):
        return []

    def nccl_all_reduce(self, timeout_s):
        return True

    def free_disk_gb(self, path, quota_gb=None):
        return 200.0


class TestAdditionalBar:
    """The three requirements that sit below the fault table."""

    def test_no_probe_terminates_on_a_single_unconfirmed_reading(self):
        """A one-tick blip resolves before the confirmation re-poll."""
        chaos = build_run()
        chaos.log_step()

        readings = iter(["crashed", "running", "running", "running"])
        original_state = chaos.wandb.state

        def flapping(run_path):
            try:
                return next(readings)
            except StopIteration:
                return original_state(run_path)

        chaos.wandb.state = flapping
        chaos.sentinel().tick()
        assert not chaos.terminated

    def test_zero_false_terminations_across_a_24h_soak(self):
        """A monitor that kills good runs gets switched off, and then there is
        no monitor at all."""
        chaos = build_run()
        sentinel = chaos.sentinel()
        watchdog = chaos.watchdog(baselines={"rollout": 200.0, "throughput": 1000.0})
        watchdog.step(0, max_steps=1440)

        for tick in range(1440):  # 24h at a 60s poll
            chaos.log_step()
            watchdog.step(tick + 1, max_steps=1440)
            watchdog.tick()
            sentinel.tick()
            chaos.clock.advance(60)
            assert not chaos.terminated, f"false termination at tick {tick}"

        assert sentinel.ticks == 1440
        fails = [a for a in chaos.alerts if a.verdict.status == "fail"]
        assert fails == [], f"false alarms on a healthy run: {[a.title() for a in fails]}"

    def test_sentinel_api_call_volume_stays_cheap(self):
        """Total sentinel cost under $1/day. Both APIs are free of charge, so the
        constraint is really call volume plus a cheap VM; this pins the volume."""
        chaos = build_run()
        chaos.log_step()
        sentinel = chaos.sentinel()

        before_runpod = len(chaos.runpod_server.calls)
        before_wandb = chaos.wandb_api.call_count
        sentinel.tick()
        per_tick = ((len(chaos.runpod_server.calls) - before_runpod)
                    + (chaos.wandb_api.call_count - before_wandb))

        calls_per_day = per_tick * (86400 / chaos.cfg.poll_interval_s)
        assert per_tick < 60, f"{per_tick} API calls per tick is too chatty"
        assert calls_per_day < 90_000, f"{calls_per_day:.0f} calls/day"

    def test_the_dead_mans_switch_is_the_backstop_for_everything_else(self):
        """SIGKILL, OOM-kill and host failure run no handler anywhere."""
        chaos = build_run()
        chaos.log_step()
        sentinel = chaos.sentinel()
        sentinel.tick()

        chaos.cut_wandb_network()
        chaos.cut_runpod_api()
        chaos.clock.advance(chaos.cfg.failsafe.dead_mans_timeout_s + 60)

        result = sentinel.tick()
        assert result.terminated
        assert any(v.probe == "sentinel.dead_mans_switch" for v in result.verdicts)

    def test_a_pod_that_never_reports_startup_ok_is_terminated(self):
        """Covers the crash W&B can never see: one before wandb.init().

        The run must have logged nothing at all. A run that is logging has
        plainly started, and the deadline check treats that as sufficient --
        terminating an actively-logging run would be self-destructive.
        """
        chaos = build_run(startup_ok=False)
        chaos.run.summary.clear()
        chaos.clock.advance(chaos.cfg.startup.deadline_s + 60)

        result = chaos.sentinel().tick()
        assert result.terminated
        assert any(v.probe == "sentinel.startup_deadline" for v in result.verdicts)

    def test_dry_run_mode_alerts_but_never_kills(self):
        """How a new config earns trust before it holds the kill switch."""
        from rlmwatch.probes import sentinel_probes

        chaos = build_run()
        levels = {p.name: Level.NOTIFY for p in sentinel_probes("rl")}
        chaos.ladder.probe_levels.update(levels)
        chaos.log_step()
        chaos.kill_9_trainer()

        chaos.sentinel().tick()
        assert not chaos.terminated
        assert chaos.alerts, "dry run must still alert"
