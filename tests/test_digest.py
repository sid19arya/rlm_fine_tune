"""Training-dynamics digest and the Hermes sink.

The digest's failure mode is not a missed alarm, it is a **misread curve** --
so most of these tests are about the digest refusing to let a reader draw a
conclusion the data cannot support.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from rlmwatch.digest import build_digest, ema, noise_band
from rlmwatch.notify import Alert, HermesSink
from rlmwatch.probes.base import Verdict
from tests.conftest import log_series


class TestNoiseBand:
    def test_it_matches_the_figure_from_the_experiment_spec(self):
        """32 rollouts at p~0.24 gives the +/-7.6 point band."""
        assert noise_band(0.24, 32) == pytest.approx(0.0755, abs=0.001)

    def test_more_rollouts_tighten_it(self):
        """128/step is why V1 can see what V0 cannot."""
        assert noise_band(0.24, 128) < noise_band(0.24, 32)
        assert noise_band(0.24, 128) == pytest.approx(0.0378, abs=0.001)

    def test_zero_rollouts_is_infinite_rather_than_a_division_error(self):
        assert noise_band(0.24, 0) == float("inf")


def test_ema_smooths_static_looking_reward():
    raw = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
    smoothed = ema(raw, span=10)
    assert len(smoothed) == len(raw)
    assert min(raw) < smoothed[-1] < max(raw)
    assert max(smoothed) - min(smoothed) < max(raw) - min(raw)


class TestTooEarly:
    """The misreading this module exists to prevent."""

    def test_flat_reward_at_step_12_is_reported_as_too_early(self, make_ctx, wandb_run):
        log_series(wandb_run, "reward/mean", [0.24] * 12)
        digest = build_digest(make_ctx(local={"step": 12, "max_steps": 20}))
        reward = next(t for t in digest.trends if t.key == "reward/mean")
        assert reward.too_early is True
        assert reward.overdue is False
        assert "too early to read" in reward.line()

    def test_the_headline_does_not_claim_a_problem_at_step_12(self, make_ctx, wandb_run):
        log_series(wandb_run, "reward/mean", [0.24] * 12)
        digest = build_digest(make_ctx(local={"step": 12, "max_steps": 20}))
        assert "overdue" not in digest.headline()

    def test_held_out_eval_is_never_expected_within_v0(self, make_ctx):
        digest = build_digest(make_ctx(local={"step": 20, "max_steps": 20}))
        eval_trend = next(t for t in digest.trends if t.key == "eval/score")
        assert eval_trend.too_early is True


class TestOverdue:
    def test_repl_error_rate_not_falling_by_step_40_is_overdue(self, make_ctx, wandb_run):
        """The spec's own diagnostic: if this is not falling by 40, something
        is broken -- bad prompt, unparseable blocks, or a rubric that does not
        discriminate."""
        log_series(wandb_run, "repl/error_rate", [0.5] * 40)
        digest = build_digest(make_ctx(local={"step": 40, "max_steps": 250}))
        errors = next(t for t in digest.trends if t.key == "repl/error_rate")
        assert errors.overdue is True
        assert "broken" in errors.note
        assert "OVERDUE" in errors.line()

    def test_a_falling_error_rate_inside_the_window_is_on_track(self, make_ctx, wandb_run):
        log_series(wandb_run, "repl/error_rate", [0.5 - 0.01 * i for i in range(20)])
        digest = build_digest(make_ctx(local={"step": 20, "max_steps": 250}))
        errors = next(t for t in digest.trends if t.key == "repl/error_rate")
        assert errors.on_track is True and errors.overdue is False
        assert errors in digest.moving

    def test_the_headline_surfaces_overdue_signals(self, make_ctx, wandb_run):
        log_series(wandb_run, "repl/error_rate", [0.5] * 40)
        digest = build_digest(make_ctx(local={"step": 40, "max_steps": 250}))
        assert "overdue" in digest.headline()


class TestDigestContent:
    def test_it_reports_spend_and_projection(self, make_ctx, wandb_run):
        log_series(wandb_run, "reward/mean", [0.24] * 5)
        digest = build_digest(make_ctx(local={"step": 5, "max_steps": 20}))
        assert digest.spend_usd is not None
        assert digest.projected_usd is not None
        assert digest.budget_usd == 5.0
        assert "Spend $" in digest.as_text()

    def test_it_reports_step_rate_and_eta(self, make_ctx, wandb_run):
        log_series(wandb_run, "reward/mean", [0.24] * 5)
        digest = build_digest(make_ctx(local={"step": 5, "max_steps": 20}))
        assert digest.seconds_per_step is not None
        assert digest.eta_hours is not None

    def test_the_reward_line_always_carries_the_noise_band(self, make_ctx, wandb_run):
        """A move smaller than the band is not a result."""
        log_series(wandb_run, "reward/mean", [0.24] * 20)
        digest = build_digest(make_ctx(local={"step": 20}), rollouts_per_step=32)
        assert digest.reward_band == pytest.approx(0.0755, abs=0.001)
        assert "noise band" in digest.as_text()

    def test_an_unreadable_run_says_so_rather_than_inventing_a_report(
        self, make_ctx, wandb_api
    ):
        wandb_api.unavailable = True
        digest = build_digest(make_ctx())
        assert digest.unavailable is not None
        assert "cannot read the run" in digest.headline()
        assert "not evidence the run is broken" in digest.as_text()

    def test_it_never_raises_on_a_run_with_no_metrics(self, make_ctx):
        digest = build_digest(make_ctx())
        assert digest.trends
        assert all(t.latest is None for t in digest.trends)

    def test_step_falls_back_to_the_wandb_summary(self, make_ctx, wandb_run):
        wandb_run.summary["_step"] = 37
        assert build_digest(make_ctx()).step == 37

    def test_serialises_for_transport(self, make_ctx, wandb_run):
        log_series(wandb_run, "reward/mean", [0.24] * 5)
        payload = build_digest(make_ctx(local={"step": 5})).as_dict()
        assert json.dumps(payload, default=str)
        assert payload["headline"] and payload["trends"]


class TestHermesSink:
    def make(self, recorder, secret="s3cret", now=lambda: 1_700_000_000.0):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.append(request)
            return httpx.Response(200, json={"ok": True})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        return HermesSink("http://hermes:8644/webhooks/rlm", secret,
                          client=client, now=now)

    def alert(self):
        return Alert(
            run_name="rlm-v0-smoke",
            verdict=Verdict(probe="rlm.reward_std", status="fail", detail="collapsed"),
            level="L3",
            spend_usd=1.2,
        )

    def test_it_signs_timestamp_dot_body_with_hmac_sha256(self):
        """Hermes Generic V2. V1 signs the body alone and has no replay
        protection -- a captured 'terminating' alert could be replayed."""
        seen = []
        sink = self.make(seen)
        assert sink.send(self.alert()) is True

        request = seen[0]
        timestamp = request.headers["X-Webhook-Timestamp"]
        expected = hmac.new(
            b"s3cret", f"{timestamp}.".encode() + request.content, hashlib.sha256
        ).hexdigest()
        assert request.headers["X-Webhook-Signature-V2"] == expected

    def test_the_signed_bytes_are_the_bytes_that_are_sent(self):
        """Re-serialising for the POST is the classic way to produce a
        signature that never verifies."""
        seen = []
        sink = self.make(seen)
        sink.post({"event_type": "rlmwatch.digest", "text": "unicode: é—"})
        request = seen[0]
        timestamp = request.headers["X-Webhook-Timestamp"]
        expected = hmac.new(
            b"s3cret", f"{timestamp}.".encode() + request.content, hashlib.sha256
        ).hexdigest()
        assert request.headers["X-Webhook-Signature-V2"] == expected

    def test_the_timestamp_is_current_for_replay_protection(self):
        seen = []
        self.make(seen).send(self.alert())
        assert seen[0].headers["X-Webhook-Timestamp"] == "1700000000"

    def test_payload_carries_an_event_type_routes_can_discriminate_on(self):
        seen = []
        self.make(seen).send(self.alert())
        payload = json.loads(seen[0].content)
        assert payload["event_type"] == "rlmwatch.fail"
        assert payload["source"] == "rlmwatch"
        assert "collapsed" in payload["text"]

    def test_it_refuses_to_construct_without_a_secret(self):
        with pytest.raises(ValueError, match="secret"):
            HermesSink("http://hermes:8644/webhooks/rlm", "")

    def test_a_rejected_webhook_reports_false_rather_than_raising(self):
        import httpx

        client = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad sig"))
        )
        sink = HermesSink("http://hermes:8644/webhooks/rlm", "s", client=client)
        assert sink.send(self.alert()) is False

    def test_a_broken_hermes_never_breaks_the_monitor(self, cfg):
        """Same rule as every other sink: a down notifier is not an outage."""
        from rlmwatch.notify import Notifier

        class Exploding(HermesSink):
            def __init__(self):
                pass

            name = "hermes"

            def send(self, alert):
                raise RuntimeError("hermes gateway down")

        notifier = Notifier(cfg, sinks=[Exploding()])
        notifier.notify(self.alert().verdict, "L1")
        assert "hermes gateway down" in notifier.last_error


def test_config_requires_a_secret_alongside_the_hermes_route():
    from rlmwatch.config import ConfigError, from_dict

    base = {"run": {"name": "x"}, "budget": {"max_usd": 5},
            "failsafe": {"on_terminal": "stop"}}
    with pytest.raises(ConfigError, match="hermes_secret"):
        from_dict({**base, "notify": {"hermes_webhook": "http://h:8644/webhooks/r"}})

    cfg = from_dict({**base, "notify": {"hermes_webhook": "http://h:8644/webhooks/r",
                                        "hermes_secret": "s"}})
    assert cfg.notify.hermes_webhook.endswith("/webhooks/r")


class TestWandbOnlySpend:
    """An observer with only a W&B key must still report honest spend."""

    def test_spend_falls_back_to_wandb_runtime(self, make_ctx, wandb_run):
        wandb_run.summary["_runtime"] = 3600.0  # 1h since wandb.init()
        ctx = make_ctx(runpod=None, local={"step": 10, "max_steps": 20})
        digest = build_digest(ctx)
        # 1h at $0.88/hr; the test config declares no storage_gb.
        assert digest.spend_usd == pytest.approx(0.88, rel=1e-3)

    def test_the_provisioning_lead_is_added_back(self, make_ctx, wandb_run):
        """_runtime starts at wandb.init(); billing started earlier."""
        wandb_run.summary["_runtime"] = 3600.0
        ctx = make_ctx(runpod=None, local={"step": 10, "max_steps": 20})
        without = build_digest(ctx).spend_usd
        with_lead = build_digest(ctx, billing_lead_s=30 * 60).spend_usd
        assert with_lead > without
        assert with_lead == pytest.approx(without * 1.5, rel=0.01)

    def test_no_runtime_means_no_invented_spend(self, make_ctx):
        ctx = make_ctx(runpod=None)
        assert build_digest(ctx).spend_usd is None


class TestHermesBriefing:
    """The briefing is an interface. If it drifts from the code, the agent
    reporting on the run is reporting the wrong thresholds."""

    def setup_method(self):
        from pathlib import Path
        self.text = (Path(__file__).resolve().parent.parent / "docs" / "HERMES.md"
                     ).read_text(encoding="utf-8")

    def test_every_tracked_metric_and_window_is_documented(self):
        from rlmwatch.digest import TRACKED

        for key, _label, (lo, hi), _wanted, _note in TRACKED:
            assert key in self.text, f"{key} is tracked in code but absent from HERMES.md"
            assert f"step {lo}-{hi}" in self.text, f"{key}'s window {lo}-{hi} is not stated"

    def test_it_states_the_noise_band_the_code_computes(self):
        from rlmwatch.digest import noise_band

        assert f"{noise_band(0.24, 32):.3f}" in self.text

    def test_it_forbids_the_agent_from_terminating(self):
        """Two systems empowered to kill a run is how a healthy run dies."""
        assert "never stop or terminate" in self.text

    def test_it_names_the_highest_value_failure(self):
        assert "reward/std" in self.text and "advantage is zero" in self.text

    def test_the_briefing_installs_the_wandb_extra(self):
        """Plain `pip install` gives a monitor that cannot read anything."""
        assert "rlmwatch[wandb]" in self.text

    def test_the_briefing_uses_the_standalone_invocation(self):
        """configs/ is not in the wheel, so -c would fail after a pip install."""
        assert "--run rlm-runpod-1/rlm-context-management/v0-smoke" in self.text
        assert "-c configs/" not in self.text
