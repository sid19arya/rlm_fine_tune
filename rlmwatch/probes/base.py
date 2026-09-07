"""Core probe interfaces.

Everything else in this library is replaceable; these four types are not.

The central design rule: **probes are pure predicates**. A probe reads state
and returns a Verdict. It never notifies, never terminates, never mutates
anything. All action lives in the escalation ladder (`rlmwatch.actions`), so
that the question "what did the monitor decide?" is always answerable by
reading one file rather than auditing every probe for side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rlmwatch.clients.runpod import RunPodClient
    from rlmwatch.clients.wandb import WandbClient
    from rlmwatch.config import RunConfig

Status = Literal["ok", "warn", "fail", "unknown"]

#: Severity order, used to combine verdicts and to compare against thresholds.
#: `unknown` sits above `ok` but below `warn`: a blind monitor is not healthy,
#: but a single blind poll is not yet an incident either. Sustained `unknown`
#: is promoted to `warn` by the escalation ladder, not here.
_SEVERITY: dict[str, int] = {"ok": 0, "unknown": 1, "warn": 2, "fail": 3}


def severity(status: Status) -> int:
    """Rank a status so verdicts can be sorted and combined."""
    return _SEVERITY[status]


def worst(statuses: list[Status]) -> Status:
    """The most severe status in a collection; `ok` for an empty collection."""
    if not statuses:
        return "ok"
    return max(statuses, key=severity)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Verdict:
    """One probe's answer at one point in time.

    `evidence` is not decoration. Every alert this library emits carries the
    raw numbers that produced it, so an operator woken at 2am can decide
    whether to intervene without opening a browser and re-deriving the state.
    """

    probe: str
    status: Status
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)
    at: datetime = field(default_factory=utcnow)

    @property
    def failed(self) -> bool:
        return self.status == "fail"

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
            "at": self.at.isoformat(),
        }

    def __str__(self) -> str:
        return f"[{self.status.upper():<7}] {self.probe}: {self.detail}"


@dataclass
class Context:
    """Everything a probe is allowed to look at.

    Probes receive this and nothing else. Anything a probe needs that is not
    here is a signal that the value belongs in config, not in probe code.
    """

    cfg: RunConfig
    wandb: WandbClient
    runpod: RunPodClient
    phase: str | None = None
    started_at: datetime = field(default_factory=utcnow)
    #: Baselines measured during the §3 warm-up: phase name -> seconds.
    #: `stall_threshold: auto` is resolved against these.
    baselines: dict[str, float] = field(default_factory=dict)
    #: Free-form state the watchdog reports from inside the process
    #: (step counter, heartbeat, current phase start time).
    local: dict[str, Any] = field(default_factory=dict)

    def elapsed_s(self) -> float:
        return (utcnow() - self.started_at).total_seconds()

    def stall_threshold_s(self, phase: str | None = None) -> float:
        """Phase-aware staleness budget.

        Flat thresholds do not work for RL: a GRPO rollout can run for many
        minutes with nothing logged, so a threshold tight enough to catch a
        hung `update` false-alarms on every normal `rollout`. `auto` resolves
        to 4x the warm-up baseline for that phase with a 600s floor.
        """
        phase = phase or self.phase
        return self.cfg.stall_threshold.resolve(phase, self.baselines)


@runtime_checkable
class Probe(Protocol):
    """A named, side-effect-free check."""

    name: str

    def check(self, ctx: Context) -> Verdict: ...


class BaseProbe:
    """Convenience base with verdict constructors.

    Subclasses implement `check`. The helpers exist so that every verdict in
    the library is built the same way and always carries its evidence.
    """

    name: str = "unnamed"

    def check(self, ctx: Context) -> Verdict:  # pragma: no cover - interface
        raise NotImplementedError

    def _verdict(self, status: Status, detail: str, **evidence: Any) -> Verdict:
        return Verdict(probe=self.name, status=status, detail=detail, evidence=evidence)

    def ok(self, detail: str, **evidence: Any) -> Verdict:
        return self._verdict("ok", detail, **evidence)

    def warn(self, detail: str, **evidence: Any) -> Verdict:
        return self._verdict("warn", detail, **evidence)

    def fail(self, detail: str, **evidence: Any) -> Verdict:
        return self._verdict("fail", detail, **evidence)

    def unknown(self, detail: str, **evidence: Any) -> Verdict:
        """Use whenever the answer could not be determined.

        Never return `ok` for "I could not tell". The distinction is what lets
        the ladder treat a monitor that has gone blind as its own incident
        instead of silently reporting health.
        """
        return self._verdict("unknown", detail, **evidence)
