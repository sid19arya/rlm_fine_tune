"""Escalation ladder, notification and diagnostics (spec section 6)."""

from __future__ import annotations

import pytest

from rlmwatch.actions import EscalationLadder, Level, install_signal_handlers
from rlmwatch.config import from_dict
from rlmwatch.diagnostics import Diagnostics, Snapshot
from rlmwatch.notify import Alert, Notifier, RecordingSink
from rlmwatch.probes.base import Verdict
from tests.conftest import POD_ID


def verdict(status="fail", probe="rlm.reward_std", detail="collapsed", **evidence):
    return Verdict(probe=probe, status=status, detail=detail, evidence=evidence)


class FakeTrainer:
    def __init__(self, *, can_checkpoint=True):
        self.can_checkpoint = can_checkpoint
        self.requested = False

    def request_checkpoint(self):
        self.requested = True

    def wait_for_checkpoint(self, timeout_s):
        return self.can_checkpoint


class FakeWandbModule:
    def __init__(self):
        self.calls = []
        self.run = None

    def alert(self, title, text):
        self.calls.append(("alert", title))

    def finish(self):
        self.calls.append(("finish", None))


@pytest.fixture
def sink():
    return RecordingSink()


@pytest.fixture
def notifier(cfg, sink):
    return Notifier(cfg, sinks=[sink])


@pytest.fixture
def make_ladder(cfg, notifier, runpod_client):
    def _make(**kwargs):
        kwargs.setdefault("notifier", notifier)
        kwargs.setdefault("runpod", runpod_client)
        kwargs.setdefault("sleep", lambda _: None)
        return EscalationLadder(kwargs.pop("cfg", cfg), **kwargs)

    return _make


class TestConfirmation:
    """No probe may terminate on a single unconfirmed reading."""

    def test_unconfirmed_fail_stops_at_notify(self, make_ladder, sink):
        ladder = make_ladder()
        record = ladder.handle(verdict(), recheck=lambda: verdict(status="ok"))
        assert record.level == Level.NOTIFY
        assert record.confirmed is False
        assert "did not reproduce" in sink.alerts[-1].verdict.detail

    def test_confirmed_fail_escalates(self, make_ladder):
        ladder = make_ladder(trainer=FakeTrainer())
        record = ladder.handle(verdict(), recheck=verdict)
        assert record.confirmed is True
        assert record.level == Level.HALT

    def test_no_recheck_available_means_unconfirmed(self, make_ladder):
        ladder = make_ladder()
        assert ladder.handle(verdict()).level == Level.NOTIFY

    def test_a_failing_recheck_is_not_confirmation(self, make_ladder):
        def boom():
            raise RuntimeError("network down")

        ladder = make_ladder()
        assert ladder.handle(verdict(), recheck=boom).level == Level.NOTIFY

    def test_confirmation_waits_the_configured_delay(self, cfg, notifier, runpod_client):
        slept = []
        ladder = EscalationLadder(cfg, notifier=notifier, runpod=runpod_client,
                                  sleep=slept.append)
        ladder.handle(verdict(), recheck=verdict)
        assert slept == [cfg.failsafe.confirm_delay_s]


class TestUnknownHandling:
    """A monitor that has gone blind is an incident, but never a kill reason."""

    def test_first_unknown_is_only_logged(self, make_ladder, sink):
        ladder = make_ladder()
        record = ladder.handle(verdict(status="unknown"))
        assert record.level == Level.LOG
        assert sink.alerts == []

    def test_sustained_unknown_is_promoted_to_warn(self, make_ladder, cfg):
        # now() is consulted once per handle(), and only on the unknown path.
        clock = iter([0.0, cfg.failsafe.unknown_tolerance_s + 1])
        ladder = make_ladder(now=lambda: next(clock))
        ladder.handle(verdict(status="unknown"))
        record = ladder.handle(verdict(status="unknown"))
        assert record.level == Level.NOTIFY
        assert record.verdict.status == "warn"
        assert record.verdict.evidence["promoted_from"] == "unknown"

    def test_sustained_unknown_never_terminates(self, make_ladder, cfg, runpod_server):
        # now() is consulted once per handle(), and only on the unknown path.
        clock = iter([0.0, cfg.failsafe.unknown_tolerance_s + 1])
        ladder = make_ladder(now=lambda: next(clock))
        ladder.handle(verdict(status="unknown"))
        ladder.handle(verdict(status="unknown"))
        assert runpod_server.stopped == []
        assert runpod_server.terminated == []

    def test_recovery_clears_the_unknown_clock(self, make_ladder, cfg):
        clock = iter([0.0, 10.0, 20.0, cfg.failsafe.unknown_tolerance_s + 100])
        ladder = make_ladder(now=lambda: next(clock))
        ladder.handle(verdict(status="unknown"))
        ladder.handle(verdict(status="ok"))
        record = ladder.handle(verdict(status="unknown"))
        assert record.verdict.status == "unknown"


class TestLevels:
    def test_warn_reaches_notify_and_no_further(self, make_ladder, sink, runpod_server):
        ladder = make_ladder()
        record = ladder.handle(verdict(status="warn"))
        assert record.level == Level.NOTIFY
        assert len(sink.alerts) == 1
        assert runpod_server.terminated == []

    def test_levels_are_cumulative(self, make_ladder, sink):
        ladder = make_ladder(diagnostics=None, trainer=FakeTrainer())
        record = ladder.handle(verdict(), recheck=verdict)
        assert record.steps[0] == "log"
        assert any(s.startswith("notify") for s in record.steps)
        assert any(s.startswith("halt") for s in record.steps)

    def test_budget_breach_goes_straight_to_terminate(self, make_ladder, runpod_server):
        ladder = make_ladder()
        record = ladder.handle(
            verdict(probe="cost.spend", detail="cap reached"),
            recheck=lambda: verdict(probe="cost.spend"),
        )
        assert record.level == Level.TERMINATE
        assert runpod_server.stopped == [POD_ID]  # on_terminal: stop in the test config

    def test_dead_pod_is_not_worth_terminating(self, make_ladder, runpod_server):
        """Nothing to kill; escalating would just add noise."""
        ladder = make_ladder()
        record = ladder.handle(
            verdict(probe="liveness.pod_status"),
            recheck=lambda: verdict(probe="liveness.pod_status"),
        )
        assert record.level == Level.NOTIFY
        assert runpod_server.terminated == []

    def test_probe_levels_are_configurable(self, make_ladder, runpod_server):
        ladder = make_ladder(probe_levels={"rlm.reward_std": Level.NOTIFY})
        record = ladder.handle(verdict(), recheck=verdict)
        assert record.level == Level.NOTIFY

    def test_unlisted_probe_defaults_to_snapshot(self, make_ladder):
        ladder = make_ladder(diagnostics=None)
        record = ladder.handle(
            verdict(probe="something.new"), recheck=lambda: verdict(probe="something.new")
        )
        assert record.level == Level.SNAPSHOT


class TestHaltAndTerminate:
    def test_halt_requests_a_checkpoint(self, make_ladder):
        trainer = FakeTrainer()
        ladder = make_ladder(trainer=trainer)
        ladder.handle(verdict(), recheck=verdict)
        assert trainer.requested is True
        assert ladder.halt_requested is True

    def test_a_trainer_too_wedged_to_checkpoint_escalates_to_terminate(
        self, make_ladder, runpod_server
    ):
        """Waiting indefinitely is most expensive exactly when it fails."""
        ladder = make_ladder(trainer=FakeTrainer(can_checkpoint=False))
        record = ladder.handle(verdict(), recheck=verdict)
        assert record.level == Level.TERMINATE
        assert "halt_timeout" in " ".join(record.steps)
        assert runpod_server.stopped == [POD_ID]

    def test_terminate_flushes_wandb_before_killing_the_pod(self, make_ladder, runpod_server):
        """Order matters: terminate first and the last minutes are lost."""
        wandb_module = FakeWandbModule()
        ladder = make_ladder(wandb_module=wandb_module)
        ladder.handle(
            verdict(probe="cost.spend"), recheck=lambda: verdict(probe="cost.spend")
        )
        assert [c[0] for c in wandb_module.calls] == ["alert", "finish"]
        assert runpod_server.stopped == [POD_ID]

    def test_on_terminal_terminate_deletes_the_pod(self, config_dict, notifier,
                                                   runpod_client, runpod_server):
        config_dict["failsafe"] = {**config_dict["failsafe"], "on_terminal": "terminate",
                                   "checkpoint_dir_is_network_volume": True}
        ladder = EscalationLadder(from_dict(config_dict), notifier=notifier,
                                  runpod=runpod_client, sleep=lambda _: None)
        ladder.handle(verdict(probe="cost.spend"), recheck=lambda: verdict(probe="cost.spend"))
        assert runpod_server.terminated == [POD_ID]

    def test_a_failed_kill_is_logged_loudly_not_swallowed(self, make_ladder, runpod_server,
                                                          caplog):
        runpod_server.read_only_key = True
        ladder = make_ladder()
        ladder.handle(verdict(probe="cost.spend"), recheck=lambda: verdict(probe="cost.spend"))
        assert ladder.terminated is False
        assert any("still billing" in r.message for r in caplog.records)


class TestShutdownGuard:
    """A clean exit does not stop billing -- the container restarts."""

    def test_normal_completion_still_terminates(self, make_ladder, runpod_server):
        ladder = make_ladder()
        ladder.shutdown("completed")
        assert runpod_server.stopped == [POD_ID]

    def test_crash_path_terminates_too(self, make_ladder, runpod_server, sink):
        ladder = make_ladder()
        ladder.shutdown("crashed: RuntimeError('CUDA OOM')")
        assert runpod_server.stopped == [POD_ID]
        assert sink.alerts[-1].verdict.status == "fail"

    def test_shutdown_checkpoints_before_terminating(self, make_ladder):
        trainer = FakeTrainer()
        ladder = make_ladder(trainer=trainer)
        ladder.shutdown("SIGTERM")
        assert trainer.requested is True

    def test_signal_handlers_are_installed_for_sigterm_and_sigint(self, make_ladder):
        class FakeSignal:
            SIGTERM, SIGINT = 15, 2

            def __init__(self):
                self.registered = []

            def signal(self, signum, handler):
                self.registered.append(signum)

        fake = FakeSignal()
        install_signal_handlers(make_ladder(), signal_module=fake)
        assert fake.registered == [15, 2]


class TestNotifications:
    def test_alert_carries_evidence_spend_and_a_wandb_link(self, notifier):
        alert = notifier.build(
            verdict(status="fail", recent=[0.0, 0.0]), "L4",
            spend_usd=4.10, projected_usd=5.50,
        )
        text = alert.as_text()
        assert "recent=[0.0, 0.0]" in text
        assert "spend $4.10" in text
        assert "projected $5.50" in text
        assert "wandb.ai" in text

    def test_slack_payload_has_blocks(self, notifier):
        payload = notifier.build(verdict(), "L2", spend_usd=1.0).as_slack_blocks()
        assert payload["blocks"][0]["type"] == "section"
        assert "rlm.reward_std" in payload["text"]

    def test_a_broken_sink_never_breaks_the_monitor(self, cfg):
        class BrokenSink:
            name = "broken"

            def send(self, alert):
                raise RuntimeError("slack is down")

        notifier = Notifier(cfg, sinks=[BrokenSink(), RecordingSink()])
        assert notifier.notify(verdict(), "L1") == 1
        assert "slack is down" in notifier.last_error

    def test_selftest_posts_a_startup_message(self, notifier, sink):
        assert notifier.selftest() is True
        assert sink.alerts[0].verdict.probe == "notify.selftest"

    def test_heartbeat_is_a_no_op_without_a_url(self, notifier):
        assert notifier.heartbeat() is False


class TestDiagnostics:
    def test_snapshot_collects_stacks_gpu_logs_and_metrics(
        self, cfg, runpod_client, wandb_client, runpod_server, wandb_run
    ):
        runpod_server.pods[POD_ID].emit("step 3 done")
        wandb_run.log({"reward/std": 0.4})

        commands = []

        def runner(cmd, timeout=20.0):
            commands.append(cmd[0])
            return f"<output of {cmd[0]}>"

        diagnostics = Diagnostics(cfg, runpod=runpod_client, wandb=wandb_client,
                                  runner=runner, pid_finder=lambda: [111, 222])
        snap = diagnostics.snapshot("test")

        assert set(snap.stacks) == {"111", "222"}
        assert "py-spy" in commands and "nvidia-smi" in commands and "dmesg" in commands
        assert snap.logs == ["[stdout] step 3 done"]
        assert snap.metrics["reward/std"] == [0.4]

    def test_one_broken_collector_does_not_lose_the_bundle(self, cfg, runpod_client,
                                                           runpod_server):
        def runner(cmd, timeout=20.0):
            if cmd[0] == "nvidia-smi":
                raise OSError("driver wedged")
            return "ok"

        runpod_server.pods[POD_ID].emit("still here")
        diagnostics = Diagnostics(cfg, runpod=runpod_client, runner=runner,
                                  pid_finder=lambda: [])
        snap = diagnostics.snapshot("test")
        assert "nvidia-smi" in snap.errors
        assert snap.logs == ["[stdout] still here"]

    def test_py_spy_dumps_are_nonblocking(self, cfg):
        """Freezing a limping trainer mid-dump can cost the checkpoint."""
        commands = []
        diagnostics = Diagnostics(cfg, runner=lambda cmd, timeout=20.0: commands.append(cmd))
        diagnostics.py_spy_dump(42)
        assert "--nonblocking" in commands[0]

    def test_log_collection_is_bounded(self, cfg, runpod_client, runpod_server):
        for i in range(2000):
            runpod_server.pods[POD_ID].emit(f"line {i}")
        diagnostics = Diagnostics(cfg, runpod=runpod_client,
                                  runner=lambda cmd, timeout=20.0: "", pid_finder=lambda: [])
        assert len(diagnostics.snapshot("test").logs) == 500

    def test_snapshot_writes_to_disk(self, cfg, tmp_path):
        snap = Snapshot(reason="hang", stacks={"1": "traceback"})
        path = snap.write(tmp_path)
        assert path.exists() and "traceback" in path.read_text(encoding="utf-8")

    def test_snapshot_is_taken_before_termination(self, cfg, notifier, runpod_client,
                                                  runpod_server):
        """Killed without a dump teaches you nothing and it will hang again."""
        order = []

        class OrderedDiagnostics(Diagnostics):
            def snapshot(self, reason, **kwargs):
                order.append("snapshot")
                return Snapshot(reason=reason)

            def upload(self, snapshot, **kwargs):
                return None

        class OrderedRunPod:
            def stop(self, pod_id):
                order.append("stop")

            def terminate(self, pod_id):
                order.append("terminate")

        ladder = EscalationLadder(
            cfg, notifier=notifier, diagnostics=OrderedDiagnostics(cfg),
            runpod=OrderedRunPod(), sleep=lambda _: None,
            probe_levels={"rlm.reward_std": Level.TERMINATE},
        )
        ladder.handle(verdict(), recheck=verdict)
        assert order == ["snapshot", "stop"]


def test_alert_renders_without_optional_fields():
    alert = Alert(run_name="r", verdict=verdict(status="warn"), level="L1")
    assert "[WARN] r: rlm.reward_std" in alert.as_text()
