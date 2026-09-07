"""Notification sinks.

One rule shapes every message here: **an operator woken at 2am must be able to
decide whether to intervene without opening a browser.** So every alert carries
the probe name, the evidence that produced the verdict, current and projected
spend, and a direct W&B link.

Sinks never raise. A monitor that crashes because Slack is down has converted a
warning into an outage, and the notification failure would itself go unreported.
Failures are recorded and surfaced through `last_error` instead.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from rlmwatch.config import RunConfig
from rlmwatch.probes.base import Verdict

log = logging.getLogger("rlmwatch.notify")

_EMOJI = {"ok": ":white_check_mark:", "warn": ":warning:", "fail": ":rotating_light:",
          "unknown": ":grey_question:"}


@dataclass
class Alert:
    """A notification payload, assembled once and rendered per sink."""

    run_name: str
    verdict: Verdict
    level: str
    spend_usd: float | None = None
    projected_usd: float | None = None
    wandb_url: str = ""
    pod_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def title(self) -> str:
        return f"[{self.verdict.status.upper()}] {self.run_name}: {self.verdict.probe}"

    def as_text(self) -> str:
        lines = [self.title(), self.verdict.detail]
        if self.verdict.evidence:
            lines.append(
                "evidence: " + ", ".join(f"{k}={v}" for k, v in self.verdict.evidence.items())
            )
        money = []
        if self.spend_usd is not None:
            money.append(f"spend ${self.spend_usd:.2f}")
        if self.projected_usd is not None:
            money.append(f"projected ${self.projected_usd:.2f}")
        if money:
            lines.append(" | ".join(money))
        lines.append(f"action: {self.level}")
        if self.pod_id:
            lines.append(f"pod: {self.pod_id}")
        if self.wandb_url:
            lines.append(self.wandb_url)
        return "\n".join(lines)

    def as_slack_blocks(self) -> dict[str, Any]:
        fields = [{"type": "mrkdwn", "text": f"*{k}*\n{v}"}
                  for k, v in list(self.verdict.evidence.items())[:8]]
        if self.spend_usd is not None:
            fields.append({"type": "mrkdwn", "text": f"*spend*\n${self.spend_usd:.2f}"})
        if self.projected_usd is not None:
            fields.append(
                {"type": "mrkdwn", "text": f"*projected*\n${self.projected_usd:.2f}"}
            )

        blocks: list[dict[str, Any]] = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{_EMOJI.get(self.verdict.status, '')} *{self.title()}*\n"
                            f"{self.verdict.detail}",
                },
            }
        ]
        if fields:
            blocks.append({"type": "section", "fields": fields})
        context = [f"action: `{self.level}`"]
        if self.pod_id:
            context.append(f"pod: `{self.pod_id}`")
        if self.wandb_url:
            context.append(f"<{self.wandb_url}|open in W&B>")
        blocks.append(
            {"type": "context", "elements": [{"type": "mrkdwn", "text": " · ".join(context)}]}
        )
        return {"text": self.title(), "blocks": blocks}

    def as_json(self) -> dict[str, Any]:
        return {
            "run": self.run_name,
            "level": self.level,
            "spend_usd": self.spend_usd,
            "projected_usd": self.projected_usd,
            "wandb_url": self.wandb_url,
            "pod_id": self.pod_id,
            **self.verdict.as_dict(),
            **self.extra,
        }


class Sink(Protocol):
    name: str

    def send(self, alert: Alert) -> bool: ...


class ConsoleSink:
    """Always available, and the only sink that works with no network."""

    name = "console"

    def __init__(self, stream=None) -> None:
        self._stream = stream or sys.stderr

    def send(self, alert: Alert) -> bool:
        print(alert.as_text(), file=self._stream, flush=True)
        return True


class WebhookSink:
    """Slack-shaped or plain JSON POST."""

    def __init__(self, url: str, *, name: str = "webhook", slack: bool = False,
                 client: httpx.Client | None = None, timeout: float = 10.0) -> None:
        self.url = url
        self.name = name
        self.slack = slack
        self._client = client or httpx.Client(timeout=timeout)

    def send(self, alert: Alert) -> bool:
        payload = alert.as_slack_blocks() if self.slack else alert.as_json()
        response = self._client.post(self.url, json=payload)
        return response.status_code < 400


class HermesSink:
    """Nous Research Hermes Agent inbound webhook.

    Hermes runs a gateway that accepts POSTs at
    ``http://<host>:8644/webhooks/<route>`` and relays to whatever channel the
    agent is configured for (Telegram, Discord, Slack, ...). We are the trigger;
    Hermes decides how to present it.

    Signed with Hermes's **Generic V2** scheme, which is the one to use: the
    HMAC-SHA256 covers ``<timestamp>.<body>`` and the receiver rejects a
    timestamp more than 300s from its own clock. V1 signs the body alone and so
    has no replay protection -- a captured "budget exceeded, terminating" alert
    could be replayed later. V2 costs nothing extra.

    Note the body is signed **exactly as sent**, so it is serialised once and
    that byte string is both signed and posted. Re-serialising for the POST is
    the classic way to produce a signature that never verifies.
    """

    def __init__(self, url: str, secret: str, *, name: str = "hermes",
                 client: httpx.Client | None = None, timeout: float = 15.0,
                 now=None) -> None:
        if not secret:
            raise ValueError(
                "HermesSink requires the shared secret (WEBHOOK_SECRET, or the "
                "per-route secret from config.yaml). An unsigned webhook is "
                "rejected by the gateway."
            )
        self.url = url
        self.name = name
        self._secret = secret.encode("utf-8")
        self._now = now or time.time
        self._client = client or httpx.Client(timeout=timeout)

    def payload(self, alert: Alert) -> dict[str, Any]:
        """What Hermes receives. `event_type` is how routes discriminate."""
        return {
            "event_type": f"rlmwatch.{alert.verdict.status}",
            "source": "rlmwatch",
            "run": alert.run_name,
            "summary": alert.title(),
            "text": alert.as_text(),
            **alert.as_json(),
        }

    def send(self, alert: Alert) -> bool:
        return self.post(self.payload(alert))

    def post(self, payload: dict[str, Any]) -> bool:
        """Sign and POST an arbitrary payload. Used for digests too."""
        body = json.dumps(payload, default=str).encode("utf-8")
        timestamp = str(int(self._now()))
        signature = hmac.new(
            self._secret,
            f"{timestamp}.".encode() + body,
            hashlib.sha256,
        ).hexdigest()
        response = self._client.post(
            self.url,
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature-V2": signature,
                "X-Webhook-Timestamp": timestamp,
            },
        )
        return response.status_code < 400


class Notifier:
    """Fan-out across sinks, plus the external heartbeat ping.

    Sink exceptions are swallowed by design -- see the module docstring. The
    return value reports how many sinks accepted the alert so callers can tell
    "notified" from "tried to notify".
    """

    def __init__(self, cfg: RunConfig, sinks: list[Sink] | None = None,
                 client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self.last_error: str | None = None
        self.sent: list[Alert] = []
        self._client = client or httpx.Client(timeout=10.0)

        if sinks is not None:
            self.sinks = list(sinks)
        else:
            self.sinks = []
            if cfg.notify.console:
                self.sinks.append(ConsoleSink())
            if cfg.notify.slack_webhook:
                self.sinks.append(
                    WebhookSink(cfg.notify.slack_webhook, name="slack", slack=True,
                                client=self._client)
                )
            if cfg.notify.webhook:
                self.sinks.append(WebhookSink(cfg.notify.webhook, client=self._client))
            if cfg.notify.hermes_webhook and cfg.notify.hermes_secret:
                self.sinks.append(
                    HermesSink(cfg.notify.hermes_webhook, cfg.notify.hermes_secret,
                               client=self._client)
                )

    def build(self, verdict: Verdict, level: str, *, spend_usd: float | None = None,
              projected_usd: float | None = None, **extra: Any) -> Alert:
        return Alert(
            run_name=self.cfg.run.name,
            verdict=verdict,
            level=level,
            spend_usd=spend_usd,
            projected_usd=projected_usd,
            wandb_url=self.cfg.run.wandb_url,
            pod_id=self.cfg.run.pod_id,
            extra=extra,
        )

    def send(self, alert: Alert) -> int:
        delivered = 0
        for sink in self.sinks:
            try:
                if sink.send(alert):
                    delivered += 1
                else:
                    self.last_error = f"{sink.name} rejected the alert"
            except Exception as exc:  # noqa: BLE001 - a sink must never break the monitor
                self.last_error = f"{sink.name}: {exc}"
                log.warning("notification sink %s failed: %s", sink.name, exc)
        self.sent.append(alert)
        return delivered

    def notify(self, verdict: Verdict, level: str, **kwargs: Any) -> int:
        return self.send(self.build(verdict, level, **kwargs))

    def heartbeat(self, *, fail: bool = False) -> bool:
        """Ping the external dead-man's-switch service (Healthchecks.io or similar).

        This is how the sentinel's *own* silence gets noticed. Nothing in this
        library watches the sentinel -- the spec is explicit that a fourth layer
        watching the third is the wrong answer, so the job is delegated outward.
        """
        url = self.cfg.notify.heartbeat_url
        if not url:
            return False
        try:
            response = self._client.post(url + ("/fail" if fail else ""), content=b"")
            return response.status_code < 400
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"heartbeat: {exc}"
            log.warning("heartbeat ping failed: %s", exc)
            return False

    def selftest(self) -> bool:
        """Post a startup message. Its absence is itself the signal.

        Called by the startup gate: a notification path that is only exercised
        during an incident is a notification path nobody knows is broken.
        """
        verdict = Verdict(
            probe="notify.selftest",
            status="ok",
            detail=f"rlmwatch attached to {self.cfg.run.name}",
            evidence={"regime": self.cfg.run.regime, "pod": self.cfg.run.pod_id or "-"},
        )
        return self.send(self.build(verdict, level="L0")) > 0

    def close(self) -> None:
        self._client.close()


class RecordingSink:
    """Test/dry-run sink that keeps alerts in memory."""

    name = "recording"

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return True

    def as_json(self) -> str:
        return json.dumps([a.as_json() for a in self.alerts], indent=2, default=str)
