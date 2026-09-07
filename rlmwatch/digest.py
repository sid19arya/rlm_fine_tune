"""Training-dynamics digest: what moved since last time.

Distinct from everything else in this library. Probes are exception-based --
they answer "is something wrong". A digest answers "what is happening", which
is a different question with a different failure mode: the danger is not a
missed alarm but a **misread curve**.

The specific misreading this module is built to prevent: looking at a flat
reward line at step 12 and concluding the run is broken. At 32 rollouts/step
with roughly binary reward at p~0.24, the standard error per step is
0.43/sqrt(32) ~ +/-7.6 points. Reward is not expected to move until step 50-150.
Behavioural metrics move far earlier because they are measured per-turn rather
than per-episode.

So every trend here carries the step window in which it is *expected* to become
visible, and is reported as "too early to tell" until then. A digest that says
"reward flat" at step 12 has told you nothing and invited a bad decision.

    REPL error rate falling        step 10-25
    Fewer max_iterations timeouts  step 15-30
    Output length stabilizing      step 10-20
    Mean REPL turns per rollout    step 20-40
    Reward mean (10-step EMA)      step 50-150
    Held-out eval delta            step 150-400
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from rlmwatch.clients.wandb import WandbUnavailable
from rlmwatch.config import RunConfig
from rlmwatch.probes.base import Context
from rlmwatch.probes.cost import elapsed_billed_hours
from rlmwatch.probes.health import read_window, trend

Direction = Literal["rising", "falling", "flat", "unknown"]

#: Canonical metric -> (label, step window in which movement is expected,
#: the direction that means progress, why it matters).
#: Windows are from the experiment spec's "what moves when" table.
TRACKED: tuple[tuple[str, str, tuple[int, int], Direction, str], ...] = (
    (
        "repl/error_rate",
        "REPL error rate",
        (10, 25),
        "falling",
        "the earliest sign of learning; if this is not falling by step 40 "
        "something is broken -- bad prompt, unparseable blocks, or a rubric "
        "that does not discriminate",
    ),
    (
        "repl/timeout_rate",
        "max_iterations timeouts",
        (15, 30),
        "falling",
        "rollouts running out of turns before answering",
    ),
    (
        "completion/length",
        "output length",
        (10, 20),
        "flat",
        "stabilising is the healthy signal; a persistent climb is length hacking",
    ),
    (
        "repl/mean_turns",
        "mean REPL turns",
        (20, 40),
        "rising",
        "falling turns while reward rises is the degeneracy kill criterion",
    ),
    (
        "repl/nontrivial_fraction",
        "non-trivial REPL ops",
        (20, 40),
        "rising",
        "the fraction doing real work (regex, split, loop, aggregate) rather "
        "than printing a truncated slice",
    ),
    (
        "reward/mean",
        "reward mean (10-step EMA)",
        (50, 150),
        "rising",
        "the last thing to move; raw per-step reward is static at this n",
    ),
    (
        "eval/score",
        "held-out eval",
        (150, 400),
        "rising",
        "the actual result; not expected within V0 at all",
    ),
)


def ema(values: list[float], span: int = 10) -> list[float]:
    """Exponential moving average. Raw per-step reward is static at n=128."""
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1 - alpha) * out[-1])
    return out


def noise_band(p: float, rollouts_per_step: int) -> float:
    """Standard error of a roughly-binary reward at `rollouts_per_step`.

    This is the number that makes a reward curve readable. At 32 rollouts and
    p~0.24 it is ~0.076, so a 5-point move is inside the noise and means
    nothing. Drawn on every reward line the digest reports.
    """
    if rollouts_per_step <= 0:
        return float("inf")
    p = min(max(p, 0.0), 1.0)
    return math.sqrt(max(p * (1 - p), 1e-9)) / math.sqrt(rollouts_per_step)


@dataclass
class Trend:
    """One tracked signal, with the context needed to read it correctly."""

    key: str
    label: str
    latest: float | None
    slope: float
    direction: Direction
    wanted: Direction
    expected_from: int
    expected_to: int
    step: int
    note: str
    samples: int = 0

    @property
    def too_early(self) -> bool:
        """Below the window where this signal is expected to become visible."""
        return self.step < self.expected_from

    @property
    def overdue(self) -> bool:
        """Past the window and still not moving the right way.

        This is the actionable state: not an alarm, but the thing a human
        should look at.
        """
        return (
            not self.too_early
            and self.step > self.expected_to
            and self.direction != self.wanted
            and self.latest is not None
        )

    @property
    def on_track(self) -> bool:
        return self.latest is not None and self.direction == self.wanted

    def line(self) -> str:
        if self.latest is None:
            return f"  {self.label}: not logged"
        value = f"{self.latest:.3f}"
        if self.too_early:
            return (f"  {self.label}: {value} -- too early to read "
                    f"(expected by step {self.expected_from}-{self.expected_to})")
        arrow = {"rising": "up", "falling": "down", "flat": "flat"}.get(self.direction, "?")
        status = "on track" if self.on_track else ("OVERDUE" if self.overdue else "flat")
        return f"  {self.label}: {value} ({arrow}, {self.slope:+.4f}/step) -- {status}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "latest": self.latest,
            "slope": self.slope,
            "direction": self.direction,
            "wanted": self.wanted,
            "expected_window": [self.expected_from, self.expected_to],
            "too_early": self.too_early,
            "overdue": self.overdue,
            "on_track": self.on_track,
            "samples": self.samples,
        }


@dataclass
class Digest:
    """One periodic report on training dynamics."""

    run_name: str
    wandb_url: str
    step: int | None
    max_steps: int | None
    trends: list[Trend] = field(default_factory=list)
    reward_ema: float | None = None
    reward_band: float | None = None
    spend_usd: float | None = None
    projected_usd: float | None = None
    budget_usd: float | None = None
    seconds_per_step: float | None = None
    eta_hours: float | None = None
    unavailable: str | None = None
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def moving(self) -> list[Trend]:
        return [t for t in self.trends if t.on_track and not t.too_early]

    @property
    def overdue(self) -> list[Trend]:
        return [t for t in self.trends if t.overdue]

    @property
    def pending(self) -> list[Trend]:
        return [t for t in self.trends if t.too_early]

    def headline(self) -> str:
        if self.unavailable:
            return f"{self.run_name}: cannot read the run ({self.unavailable})"
        step = f"step {self.step}" + (f"/{self.max_steps}" if self.max_steps else "")
        if self.overdue:
            return (f"{self.run_name} at {step}: "
                    f"{len(self.overdue)} signal(s) overdue -- worth a look")
        if self.moving:
            return f"{self.run_name} at {step}: {len(self.moving)} signal(s) moving as expected"
        return f"{self.run_name} at {step}: nothing has moved yet"

    def as_text(self) -> str:
        lines = [self.headline(), ""]

        if self.unavailable:
            lines.append(
                "The monitor could not read W&B. This is not evidence the run is "
                "broken -- it means the monitor is blind. If it stays blind, the "
                "sentinel's dead-man's switch will act on it."
            )
            return "\n".join(lines)

        if self.moving:
            lines.append("Moving:")
            lines += [t.line() for t in self.moving]
            lines.append("")
        if self.overdue:
            lines.append("Overdue -- past the window where these should have moved:")
            lines += [f"{t.line()}\n      why it matters: {t.note}" for t in self.overdue]
            lines.append("")
        if self.pending:
            lines.append("Too early to read:")
            lines += [t.line() for t in self.pending]
            lines.append("")

        if self.reward_ema is not None and self.reward_band is not None:
            lines.append(
                f"Reward 10-step EMA {self.reward_ema:.3f} "
                f"+/-{self.reward_band:.3f} noise band. A move smaller than the "
                f"band is not a result."
            )
        if self.seconds_per_step:
            eta = f", ETA {self.eta_hours:.1f}h" if self.eta_hours else ""
            lines.append(f"Step rate {self.seconds_per_step:.0f}s/step{eta}.")
        if self.spend_usd is not None:
            projected = (f", projecting ${self.projected_usd:.2f}"
                         if self.projected_usd is not None else "")
            budget = f" of ${self.budget_usd:.2f} cap" if self.budget_usd else ""
            lines.append(f"Spend ${self.spend_usd:.2f}{projected}{budget}.")
        if self.wandb_url:
            lines.append(self.wandb_url)
        return "\n".join(lines).rstrip()

    def as_dict(self) -> dict[str, Any]:
        return {
            "run": self.run_name,
            "at": self.at.isoformat(),
            "headline": self.headline(),
            "step": self.step,
            "max_steps": self.max_steps,
            "unavailable": self.unavailable,
            "reward_ema": self.reward_ema,
            "reward_noise_band": self.reward_band,
            "seconds_per_step": self.seconds_per_step,
            "eta_hours": self.eta_hours,
            "spend_usd": self.spend_usd,
            "projected_usd": self.projected_usd,
            "budget_usd": self.budget_usd,
            "wandb_url": self.wandb_url,
            "trends": [t.as_dict() for t in self.trends],
        }


def _direction(slope: float, tolerance: float) -> Direction:
    if abs(slope) <= tolerance:
        return "flat"
    return "rising" if slope > 0 else "falling"


def build_digest(ctx: Context, *, window: int = 50,
                 rollouts_per_step: int | None = None,
                 billing_lead_s: float = 0.0) -> Digest:
    """Read the run and assemble a digest. Never raises.

    An unreadable run produces a digest that says so, rather than an exception
    or -- worse -- a confident report built from missing data.

    `billing_lead_s` is how long the pod had been billing before the training
    run started -- provisioning, setup, the model download. It only matters
    when RunPod cannot be read and elapsed time has to come from W&B's own
    `_runtime`, which starts at `wandb.init()` and so understates the bill by
    exactly that much. Passing it lets a W&B-only observer report honest spend
    with no RunPod credential at all.
    """
    cfg: RunConfig = ctx.cfg
    digest = Digest(
        run_name=cfg.run.name,
        wandb_url=cfg.run.wandb_url,
        step=ctx.local.get("step"),
        max_steps=ctx.local.get("max_steps"),
        budget_usd=cfg.budget.max_usd,
    )

    try:
        summary = ctx.wandb.summary(cfg.run.wandb)
    except WandbUnavailable as exc:
        digest.unavailable = str(exc)
        return digest

    if digest.step is None:
        raw_step = summary.get("_step")
        digest.step = int(raw_step) if isinstance(raw_step, (int, float)) else None
    step = digest.step or 0

    for key, label, (lo, hi), wanted, note in TRACKED:
        try:
            values = read_window(ctx, key, window)
        except WandbUnavailable:
            values = []
        slope = trend(values) if values else 0.0
        # Tolerance scales with the metric's own magnitude: a 0.001/step drift
        # is meaningful for an error rate and noise for a token count.
        scale = abs(values[-1]) if values else 1.0
        tolerance = max(scale * 1e-3, 1e-6)
        digest.trends.append(
            Trend(
                key=key, label=label,
                latest=values[-1] if values else None,
                slope=slope,
                direction=_direction(slope, tolerance) if values else "unknown",
                wanted=wanted, expected_from=lo, expected_to=hi,
                step=step, note=note, samples=len(values),
            )
        )

    try:
        rewards = read_window(ctx, "reward/mean", window)
    except WandbUnavailable:
        rewards = []
    if rewards:
        smoothed = ema(rewards, span=10)
        digest.reward_ema = smoothed[-1]
        n = rollouts_per_step or ctx.local.get("rollouts_per_step") or 32
        digest.reward_band = noise_band(smoothed[-1], int(n))

    hours = None
    if ctx.runpod is not None:
        hours = elapsed_billed_hours(ctx)
    if hours is None:
        # W&B-only fallback, so an observer with just a W&B key can still
        # report spend. `_runtime` is seconds since wandb.init(), so the
        # provisioning lead has to be added back or the figure is optimistic --
        # and an optimistic spend number is the one you least want to trust.
        runtime = summary.get("_runtime")
        if isinstance(runtime, (int, float)):
            hours = (float(runtime) + billing_lead_s) / 3600.0
    if hours is not None:
        digest.spend_usd = cfg.budget.spend_at(hours)
        if step > 0:
            digest.seconds_per_step = hours * 3600.0 / step
            if digest.max_steps:
                remaining = max(digest.max_steps - step, 0)
                digest.eta_hours = remaining * digest.seconds_per_step / 3600.0
                total_h = hours + digest.eta_hours
                digest.projected_usd = cfg.budget.spend_at(total_h)
        if digest.projected_usd is None:
            digest.projected_usd = cfg.budget.spend_at(cfg.budget.max_wall_clock_h)

    return digest
