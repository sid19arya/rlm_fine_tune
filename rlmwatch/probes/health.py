"""Progress and metric-shape probes -- is it going anywhere? (spec section 4.2)

Regime-independent checks: loss finiteness, gradient norm, throughput against
the warm-up baseline, and projected completion against the wall-clock budget.

The regime-specific shape probes -- the ones that catch a run which is perfectly
alive and training toward a worthless model -- live in `probes/rlm.py`.
"""

from __future__ import annotations

import math

from rlmwatch.clients.wandb import WandbUnavailable
from rlmwatch.probes.base import BaseProbe, Context, Verdict


def read_window(ctx: Context, canonical_key: str, n: int, *, raw: bool = False) -> list[float]:
    """Fetch a metric window, translating the canonical name through aliases.

    Trainers name the same quantity differently (TRL, verl and OpenRLHF all
    disagree). The mapping lives in config so the trainer is never patched.
    """
    key = ctx.cfg.health.key_for(canonical_key)
    reader = ctx.wandb.raw_metric_window if raw else ctx.wandb.metric_window
    return reader(ctx.cfg.run.wandb, key, n)


def trend(values: list[float]) -> float:
    """Least-squares slope per sample. 0.0 for fewer than two points."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2
    mean_y = sum(values) / n
    numerator = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    return numerator / denominator if denominator else 0.0


class LossFiniteProbe(BaseProbe):
    """NaN/Inf loss is an immediate fail, never a warn.

    Every step after the first non-finite loss is wasted money: the weights are
    already poisoned and no amount of further training recovers them.
    """

    name = "health.loss_finite"

    def check(self, ctx: Context) -> Verdict:
        try:
            values = read_window(ctx, "train/loss", 10, raw=True)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read loss: {exc}")
        if not values:
            return self.unknown("no loss logged yet")

        bad = [v for v in values if not math.isfinite(v)]
        if bad:
            return self.fail(
                f"loss is non-finite ({bad[-1]}) -- the weights are already poisoned, "
                f"every further step is wasted spend",
                recent=values[-5:],
            )
        return self.ok(f"loss finite, latest {values[-1]:.4f}", latest=values[-1])


class GradNormProbe(BaseProbe):
    """Gradient norm inside a sane band.

    Collapse toward zero means training has silently become a no-op; explosion
    means the next step will wreck the policy. Both warn rather than fail --
    a single bad batch is normal, a sustained excursion is not, and the ladder
    handles sustain.
    """

    name = "health.grad_norm"

    def check(self, ctx: Context) -> Verdict:
        try:
            values = read_window(ctx, "train/grad_norm", 20)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read grad_norm: {exc}")
        if not values:
            return self.unknown("no grad_norm logged yet")

        latest = values[-1]
        lo, hi = ctx.cfg.health.grad_norm_min, ctx.cfg.health.grad_norm_max
        evidence = {"latest": latest, "min": lo, "max": hi, "recent": values[-5:]}
        if latest < lo:
            return self.warn(
                f"grad_norm collapsed to {latest:.2e} -- gradients this small mean "
                f"training is effectively a no-op", **evidence,
            )
        if latest > hi:
            return self.warn(f"grad_norm spiked to {latest:.2e}", **evidence)
        return self.ok(f"grad_norm {latest:.3f}", **evidence)


class ThroughputProbe(BaseProbe):
    """Sustained throughput degradation against the warm-up baseline.

    Catches thermal throttling, dataloader starvation, memory fragmentation and
    KV-cache pressure -- all of which look completely healthy on a liveness
    check while quietly doubling the cost of the run.
    """

    name = "health.throughput"

    def check(self, ctx: Context) -> Verdict:
        baseline = ctx.baselines.get("throughput")
        if not baseline:
            return self.unknown("no warm-up throughput baseline recorded")
        try:
            values = read_window(ctx, "train/throughput", 20)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read throughput: {exc}")
        if not values:
            return self.unknown("no throughput logged yet")

        recent = sum(values[-5:]) / len(values[-5:])
        drop_pct = (baseline - recent) / baseline * 100.0
        limit = ctx.cfg.health.throughput_degradation_pct
        evidence = {"recent": round(recent, 2), "baseline": round(baseline, 2),
                    "drop_pct": round(drop_pct, 1), "limit_pct": limit}
        if drop_pct > limit:
            return self.warn(
                f"throughput down {drop_pct:.0f}% from baseline "
                f"({recent:.1f} vs {baseline:.1f})", **evidence,
            )
        return self.ok(f"throughput {recent:.1f} ({-drop_pct:+.0f}% vs baseline)", **evidence)


class ProjectedCompletionProbe(BaseProbe):
    """Will this finish inside the wall-clock budget?

    Answered from the measured step rate, not the plan. The point is to find out
    at hour two that the run needs forty hours, not at hour twenty-three.
    """

    name = "health.projected_completion"

    def check(self, ctx: Context) -> Verdict:
        total = ctx.local.get("max_steps") or ctx.cfg.health.plateau_window
        step = ctx.local.get("step")
        if not step or not total:
            return self.unknown("step or max_steps unknown")

        elapsed_h = ctx.elapsed_s() / 3600.0
        if elapsed_h <= 0 or step <= 0:
            return self.unknown("not enough elapsed time to project")

        projected_h = elapsed_h / step * total
        limit_h = ctx.cfg.budget.max_wall_clock_h
        evidence = {"step": step, "max_steps": total, "elapsed_h": round(elapsed_h, 2),
                    "projected_h": round(projected_h, 2), "limit_h": limit_h}
        if projected_h > limit_h:
            return self.warn(
                f"projected {projected_h:.1f}h to finish {total} steps, budget is "
                f"{limit_h:.1f}h", **evidence,
            )
        return self.ok(f"projected {projected_h:.1f}h of {limit_h:.1f}h budget", **evidence)


HEALTH_PROBES: tuple[BaseProbe, ...] = (
    LossFiniteProbe(),
    GradNormProbe(),
    ThroughputProbe(),
    ProjectedCompletionProbe(),
)
