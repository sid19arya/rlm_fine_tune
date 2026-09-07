"""Watchdog, sentinel and CLI.

The two observers are tested for the property that motivates having two: each
sees exactly what the other cannot.
"""

from __future__ import annotations

import pytest

from rlmwatch.actions import EscalationLadder, Level
from rlmwatch.cli import build_parser, main
from rlmwatch.notify import Notifier, RecordingSink
from rlmwatch.probes.base import Verdict
from rlmwatch.sentinel import Sentinel
from rlmwatch.watchdog import Watchdog
from tests.conftest import NOW, POD_ID, RUN_PATH


@pytest.fixture
def sink():
    return RecordingSink()


@pytest.fixture
def ladder(cfg, sink, runpod_client):
    return EscalationLadder(cfg, notifier=Notifier(cfg, sinks=[sink]),
                            runpod=runpod_client, sleep=lambda _: None)


class FailingProbe:
    name = "test.always_fails"

    def check(self, ctx):
        return Verdict(probe=self.name, status="fail", detail="broken")


class ExplodingProbe:
    name = "test.explodes"

    def check(self, ctx):
        raise RuntimeError("probe bug")


class TestWatchdog:
    def test_trainer_contract_records_phase_step_and_heartbeat(self, cfg, ladder,
                                                               wandb_client, runpod_client):
        clock = iter(range(1, 1000))
        watchdog = Watchdog(cfg, ladder=ladder, wandb_client=wandb_client,
                            runpod_client=runpod_client, now=lambda: float(next(clock)))
        watchdog.phase("rollout")
        watchdog.heartbeat()
        watchdog.step(7, max_steps=20)

        ctx = watchdog.context
        assert ctx.phase == "rollout"
        assert ctx.local["step"] == 7
        assert ctx.local["max_steps"] == 20
        assert ctx.local["heartbeat"] == 1

    def test_phase_changes_the_effective_stall_threshold(self, cfg, ladder, wandb_client,
                                                          runpod_client):
        watchdog = Watchdog(cfg, ladder=ladder, wandb_client=wandb_client,
                            runpod_client=runpod_client, baselines={"rollout": 200.0})
        assert watchdog.context.stall_threshold_s() == 300  # default
        watchdog.phase("rollout")
        assert watchdog.context.stall_threshold_s() == 800.0

    def test_tick_runs_probes_and_drives_the_ladder(self, cfg, ladder, wandb_client,
                                                    runpod_client, sink):
        watchdog = Watchdog(cfg, ladder=ladder, wandb_client=wandb_client,
                            runpod_client=runpod_client, probes=(FailingProbe(),))
        verdicts = watchdog.tick()
        assert [v.status for v in verdicts] == ["fail"]
        assert sink.alerts, "a confirmed fail must reach at least L1"

    def test_a_probe_that_raises_blinds_but_never_kills(self, cfg, ladder, wandb_client,
                                                        runpod_client, runpod_server):
        watchdog = Watchdog(cfg, ladder=ladder, wandb_client=wandb_client,
                            runpod_client=runpod_client, probes=(ExplodingProbe(),))
        assert watchdog.tick()[0].status == "unknown"
        assert runpod_server.stopped == [] and runpod_server.terminated == []

    def test_watchdog_probe_set_cannot_report_pod_death(self, cfg, ladder, wandb_client,
                                                        runpod_client):
        """It dies with the pod; claiming otherwise would be false assurance."""
        watchdog = Watchdog(cfg, ladder=ladder, wandb_client=wandb_client,
                            runpod_client=runpod_client)
        names = {p.name for p in watchdog.probes}
        assert "liveness.pod_status" not in names
        assert "liveness.observer_agreement" not in names

    def test_generation_backend_health_is_collected_each_tick(self, cfg, ladder,
                                                              wandb_client, runpod_client):
        watchdog = Watchdog(cfg, ladder=ladder, wandb_client=wandb_client,
                            runpod_client=runpod_client, probes=(),
                            gen_backend_health=lambda: False)
        watchdog.tick()
        assert watchdog.context.local["gen_backend_healthy"] is False

    def test_an_unreachable_health_endpoint_is_none_not_false(self, cfg, ladder,
                                                              wandb_client, runpod_client):
        """Unreachable is 'cannot tell', which is not the same as 'dead'."""
        def boom():
            raise OSError("connection refused")

        watchdog = Watchdog(cfg, ladder=ladder, wandb_client=wandb_client,
                            runpod_client=runpod_client, probes=(),
                            gen_backend_health=boom)
        watchdog.tick()
        assert watchdog.context.local["gen_backend_healthy"] is None

    def test_context_manager_terminates_on_normal_completion(self, cfg, ladder,
                                                             wandb_client, runpod_client,
                                                             runpod_server):
        """A clean exit does not stop billing; the container restarts."""
        with Watchdog(cfg, ladder=ladder, wandb_client=wandb_client,
                      runpod_client=runpod_client, probes=()):
            pass
        assert runpod_server.stopped == [POD_ID]

    def test_context_manager_terminates_on_exception_and_reraises(self, cfg, ladder,
                                                                  wandb_client,
                                                                  runpod_client,
                                                                  runpod_server, sink):
        with pytest.raises(RuntimeError, match="CUDA OOM"):
            with Watchdog(cfg, ladder=ladder, wandb_client=wandb_client,
                          runpod_client=runpod_client, probes=()):
                raise RuntimeError("CUDA OOM")
        assert runpod_server.stopped == [POD_ID]
        assert "crashed" in sink.alerts[-1].verdict.detail

    def test_halted_exposes_the_ladders_decision_to_the_training_loop(self, cfg, ladder,
                                                                      wandb_client,
                                                                      runpod_client):
        watchdog = Watchdog(cfg, ladder=ladder, wandb_client=wandb_client,
                            runpod_client=runpod_client, probes=())
        assert watchdog.halted is False
        ladder.halt_requested = True
        assert watchdog.halted is True

    def test_thread_starts_and_stops_cleanly(self, cfg, ladder, wandb_client,
                                             runpod_client):
        watchdog = Watchdog(cfg, ladder=ladder, wandb_client=wandb_client,
                            runpod_client=runpod_client, probes=(), interval_s=0.01)
        watchdog.start()
        watchdog.stop()
        assert watchdog._thread is None


class TestSentinel:
    def make(self, cfg, ladder, wandb_client, runpod_client, **kwargs):
        kwargs.setdefault("probes", ())
        kwargs.setdefault("now", lambda: NOW)
        kwargs.setdefault("sleep", lambda _: None)
        kwargs.setdefault("pod_created_at", NOW)
        return Sentinel(cfg, ladder=ladder, wandb_client=wandb_client,
                        runpod_client=runpod_client, **kwargs)

    def test_startup_deadline_not_yet_reached_is_quiet(self, cfg, ladder, wandb_client,
                                                       runpod_client):
        sentinel = self.make(cfg, ladder, wandb_client, runpod_client)
        assert sentinel.check_startup_deadline() is None

    def test_missing_startup_ok_past_the_deadline_fails(self, cfg, ladder, wandb_client,
                                                        runpod_client):
        """The only way to catch a crash before wandb.init()."""
        late = NOW + cfg.startup.deadline_s + 60
        sentinel = self.make(cfg, ladder, wandb_client, runpod_client,
                             now=lambda: late, pod_created_at=NOW)
        verdict = sentinel.check_startup_deadline()
        assert verdict is not None and verdict.status == "fail"
        assert "before wandb.init()" in verdict.detail

    def test_startup_ok_in_the_summary_satisfies_the_deadline(self, cfg, ladder,
                                                              wandb_client, runpod_client,
                                                              wandb_run):
        wandb_run.summary["startup_ok"] = 1
        late = NOW + cfg.startup.deadline_s + 60
        sentinel = self.make(cfg, ladder, wandb_client, runpod_client, now=lambda: late)
        assert sentinel.check_startup_deadline() is None
        assert sentinel.startup_ok_seen is True

    def test_a_pod_that_never_started_is_terminated(self, cfg, ladder, wandb_client,
                                                    runpod_client, runpod_server):
        late = NOW + cfg.startup.deadline_s + 60
        sentinel = self.make(cfg, ladder, wandb_client, runpod_client, now=lambda: late)
        result = sentinel.tick()
        assert result.terminated is True
        assert runpod_server.stopped == [POD_ID]

    def test_dead_mans_switch_fires_only_after_total_silence(self, cfg, ladder,
                                                             wandb_client, runpod_client,
                                                             wandb_api, runpod_server,
                                                             wandb_run):
        """The backstop for SIGKILL, OOM-kill and host failure."""
        wandb_run.summary["startup_ok"] = 1
        clock = {"t": NOW}
        sentinel = self.make(cfg, ladder, wandb_client, runpod_client,
                             now=lambda: clock["t"], probes=())
        sentinel.tick()  # establishes contact

        # Both observers go dark, and stay dark past the timeout.
        wandb_api.unavailable = True
        runpod_server.transient_failures = 10_000
        clock["t"] = NOW + cfg.failsafe.dead_mans_timeout_s + 60
        result = sentinel.tick()
        assert result.terminated is True
        assert any(v.probe == "sentinel.dead_mans_switch" for v in result.verdicts)

    def test_a_reachable_but_sick_run_does_not_trip_the_dead_mans_switch(
        self, cfg, ladder, wandb_client, runpod_client, wandb_run
    ):
        """Seeing a sick run is still seeing. The switch is for silence."""
        wandb_run.summary["startup_ok"] = 1
        clock = {"t": NOW}
        sentinel = self.make(cfg, ladder, wandb_client, runpod_client,
                             now=lambda: clock["t"], probes=(FailingProbe(),))
        sentinel.tick()
        clock["t"] = NOW + cfg.failsafe.dead_mans_timeout_s + 60
        result = sentinel.tick()
        assert not any(v.probe == "sentinel.dead_mans_switch" for v in result.verdicts)

    def test_every_tick_reports_current_and_projected_spend(self, cfg, ladder,
                                                            wandb_client, runpod_client,
                                                            wandb_run):
        wandb_run.summary["startup_ok"] = 1
        sentinel = self.make(cfg, ladder, wandb_client, runpod_client)
        result = sentinel.tick()
        assert result.spend_usd == pytest.approx(0.88)  # 1h uptime in the fake
        assert result.projected_usd == pytest.approx(2.64)  # 3h wall-clock budget

    def test_finished_run_with_a_live_pod_keeps_the_sentinel_watching(
        self, cfg, ladder, wandb_client, runpod_client, wandb_run
    ):
        """A finished run with a running pod is the reason to keep watching."""
        wandb_run.summary["startup_ok"] = 1
        wandb_run.state = "finished"
        sentinel = self.make(cfg, ladder, wandb_client, runpod_client)
        assert sentinel._finished() is False

    def test_finished_run_with_a_dead_pod_stops_the_sentinel(self, cfg, ladder,
                                                             wandb_client, runpod_client,
                                                             wandb_run, runpod_server):
        wandb_run.summary["startup_ok"] = 1
        wandb_run.state = "finished"
        runpod_server.pods[POD_ID].state = "EXITED"
        sentinel = self.make(cfg, ladder, wandb_client, runpod_client)
        assert sentinel._finished() is True

    def test_run_honours_max_ticks_for_cron_mode(self, cfg, ladder, wandb_client,
                                                 runpod_client, wandb_run):
        wandb_run.summary["startup_ok"] = 1
        sentinel = self.make(cfg, ladder, wandb_client, runpod_client)
        assert len(sentinel.run(max_ticks=3)) == 3

    def test_sentinel_probe_set_includes_what_the_watchdog_cannot_see(self, cfg, ladder,
                                                                      wandb_client,
                                                                      runpod_client):
        sentinel = Sentinel(cfg, ladder=ladder, wandb_client=wandb_client,
                            runpod_client=runpod_client)
        names = {p.name for p in sentinel.probes}
        assert "liveness.pod_status" in names
        assert "liveness.observer_agreement" in names
        assert "cost.spend" in names, "caps must be enforced from outside the pod"


class TestCli:
    def test_kill_refuses_terminate_without_explicit_confirmation(self, tmp_path,
                                                                  monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_API_KEY", "k")
        config = tmp_path / "cfg.yaml"
        config.write_text(
            "run:\n  name: r\n  pod_id: pod-1\n"
            "budget:\n  max_usd: 5\n  hourly_rate_usd: 0.88\n"
            "failsafe:\n  on_terminal: stop\n",
            encoding="utf-8",
        )
        code = main(["kill", "-c", str(config), "--action", "terminate"])
        assert code == 2
        assert "deletes pod" in capsys.readouterr().err

    def test_config_error_exits_cleanly(self, tmp_path, capsys):
        config = tmp_path / "bad.yaml"
        config.write_text("run:\n  name: r\n  regime: nonsense\n", encoding="utf-8")
        assert main(["preflight", "-c", str(config)]) == 2
        assert "config error" in capsys.readouterr().err

    def test_watch_has_a_dry_run_that_cannot_kill(self):
        args = build_parser().parse_args(["watch", "-c", "x.yaml", "--dry-run", "--once"])
        assert args.dry_run is True and args.once is True

    def test_every_subcommand_requires_a_config(self):
        parser = build_parser()
        for command in ("preflight", "watch", "snapshot", "kill"):
            with pytest.raises(SystemExit):
                parser.parse_args([command])


def test_dry_run_sentinel_caps_the_ladder_at_notify(cfg, sink, runpod_client):
    """Earn confidence in a config before it holds the kill switch."""
    from rlmwatch.probes import sentinel_probes

    levels = {p.name: Level.NOTIFY for p in sentinel_probes(cfg.run.regime)}
    ladder = EscalationLadder(cfg, notifier=Notifier(cfg, sinks=[sink]), runpod=None,
                              probe_levels=levels, sleep=lambda _: None)
    record = ladder.handle(
        Verdict(probe="cost.spend", status="fail", detail="cap"),
        recheck=lambda: Verdict(probe="cost.spend", status="fail", detail="cap"),
    )
    assert record.level == Level.NOTIFY
    assert ladder.terminated is False
    assert RUN_PATH  # config fixture sanity
