"""Cost probes (spec section 4.3).

Billing is per-second from the moment the pod is **provisioned**, not from when
the workload starts, and it does not stop when the process exits. So elapsed
time here is measured from pod creation, and the caps are enforced by the
sentinel -- they have to survive the pod becoming unresponsive, which is exactly
the state in which they matter most.

Every alert this library emits carries the projected total. An operator deciding
whether to intervene at 2am needs the number in the message, not a link.
"""

from __future__ import annotations

from rlmwatch.probes.base import BaseProbe, Context, Verdict


def elapsed_billed_hours(ctx: Context) -> float | None:
    """Hours billed so far, preferring the pod's own uptime.

    Falls back to the monitor's start time, which understates the bill: the pod
    was provisioned before the training script began. The fallback is used only
    when the RunPod API cannot be read.
    """
    pod_id = ctx.cfg.run.pod_id
    if pod_id:
        try:
            return ctx.runpod.status(pod_id).uptime_s / 3600.0
        except Exception:  # noqa: BLE001 - fall through to the local estimate
            pass
    elapsed = ctx.elapsed_s()
    return elapsed / 3600.0 if elapsed > 0 else None


class SpendProbe(BaseProbe):
    """Accrued spend against the cap. Warn at `warn_pct`, fail at the cap.

    The cap is unconditional -- there is no configuration that disables it.
    """

    name = "cost.spend"

    def check(self, ctx: Context) -> Verdict:
        hours = elapsed_billed_hours(ctx)
        if hours is None:
            return self.unknown("cannot determine elapsed billed time")

        budget = ctx.cfg.budget
        spend = budget.spend_at(hours)
        pct = spend / budget.max_usd * 100.0 if budget.max_usd else 0.0
        evidence = {
            "spend_usd": round(spend, 2),
            "max_usd": budget.max_usd,
            "pct_of_cap": round(pct, 1),
            "billed_hours": round(hours, 2),
            "hourly_rate_usd": budget.hourly_rate_usd,
        }

        if spend >= budget.max_usd:
            return self.fail(
                f"spend ${spend:.2f} has reached the ${budget.max_usd:.2f} cap "
                f"after {hours:.1f}h", **evidence,
            )
        if pct >= budget.warn_pct:
            return self.warn(
                f"spend ${spend:.2f} is {pct:.0f}% of the ${budget.max_usd:.2f} cap",
                **evidence,
            )
        return self.ok(f"spend ${spend:.2f} of ${budget.max_usd:.2f} ({pct:.0f}%)", **evidence)


class WallClockProbe(BaseProbe):
    """Hard wall-clock cap, independent of spend.

    A cheap pod left running for a week is still a bill nobody meant to pay, and
    a run that has exceeded its planned duration has usually gone wrong in a way
    the metric probes did not name.
    """

    name = "cost.wall_clock"

    def check(self, ctx: Context) -> Verdict:
        hours = elapsed_billed_hours(ctx)
        if hours is None:
            return self.unknown("cannot determine elapsed billed time")

        limit = ctx.cfg.budget.max_wall_clock_h
        evidence = {"elapsed_h": round(hours, 2), "limit_h": limit}
        if hours >= limit:
            return self.fail(f"wall clock {hours:.1f}h exceeds the {limit:.1f}h cap", **evidence)
        if hours >= limit * 0.9:
            return self.warn(f"wall clock {hours:.1f}h of {limit:.1f}h", **evidence)
        return self.ok(f"wall clock {hours:.1f}h of {limit:.1f}h", **evidence)


class CostPerProgressProbe(BaseProbe):
    """Dollars per 1% of planned steps, versus the rate set early in the run.

    Catches slow degradation that no absolute threshold sees: the run is alive,
    under budget, under wall clock -- and each percent of progress now costs
    three times what the first ten did. Rising sharply means you are paying more
    for less.
    """

    name = "cost.cost_per_progress"

    def check(self, ctx: Context) -> Verdict:
        step = ctx.local.get("step")
        total = ctx.local.get("max_steps")
        if not step or not total:
            return self.unknown("step or max_steps unknown")

        hours = elapsed_billed_hours(ctx)
        if hours is None:
            return self.unknown("cannot determine elapsed billed time")

        progress_pct = step / total * 100.0
        if progress_pct <= 0:
            return self.unknown("no measurable progress yet")

        spend = ctx.cfg.budget.spend_at(hours)
        rate = spend / progress_pct
        projected_total = rate * 100.0
        baseline = ctx.baselines.get("cost_per_pct")
        evidence = {
            "usd_per_pct": round(rate, 4),
            "progress_pct": round(progress_pct, 1),
            "spend_usd": round(spend, 2),
            "projected_total_usd": round(projected_total, 2),
            "max_usd": ctx.cfg.budget.max_usd,
        }
        if baseline:
            evidence["baseline_usd_per_pct"] = round(baseline, 4)
            evidence["ratio"] = round(rate / baseline, 2)

        if projected_total > ctx.cfg.budget.max_usd:
            return self.warn(
                f"at ${rate:.3f} per 1% of steps this run projects to "
                f"${projected_total:.2f}, over the ${ctx.cfg.budget.max_usd:.2f} cap",
                **evidence,
            )
        if baseline and rate > baseline * 2.0:
            return self.warn(
                f"cost per 1% of progress has {rate / baseline:.1f}x'd since the "
                f"baseline -- paying more for less", **evidence,
            )
        return self.ok(
            f"${rate:.3f} per 1% of steps, projecting ${projected_total:.2f}", **evidence
        )


COST_PROBES: tuple[BaseProbe, ...] = (
    SpendProbe(),
    WallClockProbe(),
    CostPerProgressProbe(),
)
