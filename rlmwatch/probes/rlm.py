"""RLM fine-tuning probes (spec section 5).

`regime: sft | rl` in config selects the probe set.

The distinction these probes exist for: **stalled is not the same as
degenerate.** Liveness monitoring catches stalled. It gives false confidence
about degenerate, which is RL's characteristic failure -- full GPU utilisation,
a steady step rate, a healthy-looking loss curve, and a policy collapsing into
reward hacking the whole time. Only the shape probes below see that.

The highest-value probe in the library is `RewardStdProbe`. If every completion
in a GRPO group scores identically, the advantage is zero, the gradient is zero,
and training is a no-op -- while every dashboard looks completely normal and the
bill accrues at the full rate.
"""

from __future__ import annotations

import statistics

from rlmwatch.clients.wandb import WandbUnavailable
from rlmwatch.probes.base import BaseProbe, Context, Verdict
from rlmwatch.probes.health import read_window, trend

# --- regime: rl ---------------------------------------------------------------


class RewardStdProbe(BaseProbe):
    """Zero-variance collapse within the GRPO group.

    Identical scores across a group mean zero advantage and zero gradient. The
    run keeps stepping, the GPUs stay busy, the loss curve stays plausible, and
    nothing is being learned. Patience is measured in steps, not seconds: an
    occasional degenerate batch is normal, a sustained run of them is not.
    """

    name = "rlm.reward_std"

    def check(self, ctx: Context) -> Verdict:
        patience = ctx.cfg.health.reward_std_patience
        floor = ctx.cfg.health.reward_std_min
        try:
            values = read_window(ctx, "reward/std", patience)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read reward/std: {exc}")
        if not values:
            return self.unknown("no reward/std logged yet")

        collapsed = [v for v in values if v < floor]
        evidence = {"recent": [round(v, 4) for v in values[-5:]], "min": floor,
                    "patience_steps": patience, "collapsed_of_window": len(collapsed),
                    "window": len(values)}

        if len(values) >= patience and len(collapsed) == len(values):
            return self.fail(
                f"reward std below {floor} for {len(values)} consecutive steps -- the "
                f"advantage is zero, so the gradient is zero and training is a no-op "
                f"while every dashboard looks normal",
                **evidence,
            )
        if collapsed and values[-1] < floor:
            return self.warn(
                f"reward std at {values[-1]:.4f}, below {floor} for the last "
                f"{len(collapsed)} of {len(values)} steps", **evidence,
            )
        return self.ok(f"reward std {values[-1]:.4f}", **evidence)


class KLDivergenceProbe(BaseProbe):
    """KL against the reference policy.

    Blowup means the policy has left the reference distribution and outputs
    degenerate. Both the level and the slope matter: a KL climbing steeply
    toward the limit is already lost, it just has not arrived yet.
    """

    name = "rlm.kl_ref"

    def check(self, ctx: Context) -> Verdict:
        limit = ctx.cfg.health.kl_max
        try:
            values = read_window(ctx, "kl/ref", 20)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read kl/ref: {exc}")
        if not values:
            return self.unknown("no kl/ref logged yet")

        latest, slope = values[-1], trend(values)
        evidence = {"latest": round(latest, 4), "max": limit, "slope_per_step": round(slope, 5),
                    "recent": [round(v, 4) for v in values[-5:]]}
        if latest > limit:
            return self.fail(
                f"KL to reference is {latest:.3f}, past the {limit} limit -- the policy "
                f"has left the reference distribution", **evidence,
            )
        if slope > 0 and len(values) >= 5 and latest + slope * 10 > limit:
            return self.warn(
                f"KL {latest:.3f} rising at {slope:.4f}/step, crosses {limit} within "
                f"~10 steps", **evidence,
            )
        return self.ok(f"KL {latest:.3f} of {limit}", **evidence)


class EntropyProbe(BaseProbe):
    """Entropy collapse: a deterministic policy has stopped exploring.

    Warns on a monotone decline before it fails on the floor, because once
    exploration is dead the remaining steps buy nothing.
    """

    name = "rlm.entropy"

    def check(self, ctx: Context) -> Verdict:
        floor = ctx.cfg.health.entropy_min
        try:
            values = read_window(ctx, "policy/entropy", 20)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read policy/entropy: {exc}")
        if not values:
            return self.unknown("no policy/entropy logged yet")

        latest, slope = values[-1], trend(values)
        evidence = {"latest": round(latest, 4), "min": floor, "slope_per_step": round(slope, 5),
                    "recent": [round(v, 4) for v in values[-5:]]}
        if latest < floor:
            return self.fail(
                f"entropy {latest:.3f} below {floor} -- exploration is dead and the "
                f"policy is effectively deterministic", **evidence,
            )
        if len(values) >= 10 and slope < 0 and all(
            b <= a for a, b in zip(values[-10:], values[-9:], strict=False)
        ):
            return self.warn(
                f"entropy declining monotonically over 10 steps to {latest:.3f}", **evidence
            )
        return self.ok(f"entropy {latest:.3f}", **evidence)


class CompletionLengthProbe(BaseProbe):
    """Length hacking, or collapse to trivial answers.

    Padding to farm reward and answering in three tokens are opposite
    pathologies with the same cause, so both ends of the band are checked.
    """

    name = "rlm.completion_length"

    def check(self, ctx: Context) -> Verdict:
        lo, hi = ctx.cfg.health.completion_len
        try:
            values = read_window(ctx, "completion/length", 20)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read completion/length: {exc}")
        if not values:
            return self.unknown("no completion/length logged yet")

        latest = values[-1]
        evidence = {"latest": round(latest, 1), "min": lo, "max": hi,
                    "slope_per_step": round(trend(values), 3)}
        if latest > hi:
            return self.warn(
                f"mean completion length {latest:.0f} exceeds {hi} -- possible length "
                f"hacking, or truncation is now discarding conclusions", **evidence,
            )
        if latest < lo:
            return self.warn(
                f"mean completion length {latest:.0f} below {lo} -- the policy may have "
                f"collapsed to trivial answers", **evidence,
            )
        return self.ok(f"completion length {latest:.0f}", **evidence)


class FormatSuccessProbe(BaseProbe):
    """Fraction of rollouts the scorer could actually parse.

    The silent-zero failure: the model stops emitting valid delimiters or
    tool-call syntax, the scorer returns zeros for everything, and reward looks
    like a smooth decline rather than a broken harness. For the RLM experiment
    this is also what catches a leftover `llm_query` reference killing every
    rollout with a NameError while the run looks healthy from outside.
    """

    name = "rlm.format_success"

    def check(self, ctx: Context) -> Verdict:
        floor = ctx.cfg.health.format_success_min
        try:
            values = read_window(ctx, "format/success_rate", 10)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read format/success_rate: {exc}")
        if not values:
            return self.unknown("no format/success_rate logged yet")

        latest = values[-1]
        evidence = {"latest": round(latest, 3), "min": floor,
                    "recent": [round(v, 3) for v in values[-5:]]}
        if latest < floor:
            return self.fail(
                f"only {latest:.0%} of rollouts parsed (floor {floor:.0%}) -- the scorer "
                f"is returning zeros for unparseable output, not measuring quality",
                **evidence,
            )
        return self.ok(f"format success {latest:.0%}", **evidence)


class ClipFractionProbe(BaseProbe):
    """PPO clip fraction. Sustained high means updates are being discarded."""

    name = "rlm.clip_fraction"

    def check(self, ctx: Context) -> Verdict:
        limit = ctx.cfg.health.clip_max
        try:
            values = read_window(ctx, "clip_fraction", 20)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read clip_fraction: {exc}")
        if not values:
            return self.unknown("no clip_fraction logged yet (not a PPO run?)")

        latest = values[-1]
        evidence = {"latest": round(latest, 3), "max": limit}
        if latest > limit:
            return self.warn(
                f"clip fraction {latest:.2f} over {limit} -- steps are too large and "
                f"updates are being thrown away", **evidence,
            )
        return self.ok(f"clip fraction {latest:.2f}", **evidence)


class RolloutDurationProbe(BaseProbe):
    """Rollout time against the warm-up baseline. The most common RL hang.

    A generation backend that is degrading rather than dead produces a run that
    passes every liveness check while its cost per step quietly triples.
    """

    name = "rlm.rollout_duration"

    def check(self, ctx: Context) -> Verdict:
        baseline = ctx.baselines.get("rollout")
        if not baseline:
            return self.unknown("no warm-up rollout baseline recorded")
        try:
            values = read_window(ctx, "rollout/duration_s", 10)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read rollout/duration_s: {exc}")
        if not values:
            return self.unknown("no rollout/duration_s logged yet")

        latest = values[-1]
        multiplier = ctx.cfg.health.rollout_duration_multiplier
        ratio = latest / baseline
        evidence = {"latest_s": round(latest, 1), "baseline_s": round(baseline, 1),
                    "ratio": round(ratio, 2), "limit_ratio": multiplier}
        if ratio > multiplier:
            return self.fail(
                f"rollout took {latest:.0f}s, {ratio:.1f}x the {baseline:.0f}s baseline -- "
                f"the generation backend is wedged or badly degraded", **evidence,
            )
        if ratio > multiplier * 0.6:
            return self.warn(f"rollout {ratio:.1f}x baseline ({latest:.0f}s)", **evidence)
        return self.ok(f"rollout {latest:.0f}s ({ratio:.1f}x baseline)", **evidence)


class GenerationBackendProbe(BaseProbe):
    """vLLM/SGLang health endpoint.

    The trainer waits on the generation backend forever if it dies. Without this
    probe that shows up only as a stall, hours later, with no cause attached.
    The watchdog supplies the result of the health check in `ctx.local`, since
    the endpoint is reachable only from inside the pod.
    """

    name = "rlm.generation_backend"

    def check(self, ctx: Context) -> Verdict:
        healthy = ctx.local.get("gen_backend_healthy")
        if healthy is None:
            return self.unknown("generation backend health not reported")
        endpoint = ctx.local.get("gen_backend_url", "")
        if not healthy:
            return self.fail(
                f"generation backend at {endpoint or 'the configured endpoint'} is not "
                f"responding -- the trainer will wait on it indefinitely",
                endpoint=endpoint,
            )
        return self.ok("generation backend healthy", endpoint=endpoint)


class VramHeadroomProbe(BaseProbe):
    """Free VRAM trending toward zero.

    RL's OOM does not arrive at step 0. Policy + reference + a KV cache that
    grows as sequences lengthen means the failure lands hours in, which is why
    this warns on the trend rather than on the level.
    """

    name = "rlm.vram_headroom"

    def check(self, ctx: Context) -> Verdict:
        run_path = ctx.cfg.run.wandb
        if not run_path:
            return self.unknown("no W&B run configured")
        try:
            free_pct = ctx.wandb.system_metrics(run_path).min_free_mem_pct()
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read GPU memory: {exc}")
        if free_pct is None:
            return self.unknown("no GPU memory metrics reported yet")

        evidence = {"min_free_pct": round(free_pct, 1)}
        if free_pct < 5.0:
            return self.warn(
                f"only {free_pct:.1f}% VRAM free -- OOM in RL arrives late, as sequences "
                f"lengthen, not at step 0", **evidence,
            )
        if free_pct < 15.0:
            return self.warn(f"VRAM headroom down to {free_pct:.1f}%", **evidence)
        return self.ok(f"VRAM headroom {free_pct:.1f}%", **evidence)


class RewardModelLatencyProbe(BaseProbe):
    """External scorer errors and latency.

    A rate-limited or half-down reward model does not raise -- it returns
    degraded scores, and the run trains on them.
    """

    name = "rlm.reward_model"

    def check(self, ctx: Context) -> Verdict:
        try:
            errors = read_window(ctx, "reward_model/error_rate", 10)
            latency = read_window(ctx, "reward_model/p99_latency_s", 10)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read reward model metrics: {exc}")
        if not errors and not latency:
            return self.unknown("no reward model metrics logged (no external scorer?)")

        error_rate = errors[-1] if errors else 0.0
        p99 = latency[-1] if latency else 0.0
        evidence = {"error_rate": round(error_rate, 3), "p99_latency_s": round(p99, 2)}
        if error_rate > 0.05:
            return self.fail(
                f"reward model error rate {error_rate:.1%} -- rewards are silently "
                f"degrading and the policy is training on them", **evidence,
            )
        if error_rate > 0.01:
            return self.warn(f"reward model error rate {error_rate:.1%}", **evidence)
        return self.ok(f"reward model healthy (p99 {p99:.1f}s)", **evidence)


class RewardHackingProbe(BaseProbe):
    """Reward climbing while the policy drifts and held-out quality does not.

    No single metric shows this; it is the conjunction that matters, which is
    why the probe reads three at once.
    """

    name = "rlm.reward_hacking"

    def check(self, ctx: Context) -> Verdict:
        try:
            reward = read_window(ctx, "reward/mean", 20)
            kl = read_window(ctx, "kl/ref", 20)
            evals = read_window(ctx, "eval/score", 5)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read reward-hacking inputs: {exc}")
        if len(reward) < 10 or len(kl) < 10:
            return self.unknown("not enough history to judge reward hacking")

        reward_slope, kl_slope = trend(reward), trend(kl)
        eval_slope = trend(evals) if len(evals) >= 3 else None
        evidence = {"reward_slope": round(reward_slope, 5), "kl_slope": round(kl_slope, 5),
                    "eval_slope": None if eval_slope is None else round(eval_slope, 5)}

        if reward_slope > 0 and kl_slope > 0 and eval_slope is not None and eval_slope <= 0:
            return self.warn(
                f"reward rising ({reward_slope:+.4f}/step) and KL rising "
                f"({kl_slope:+.4f}/step) while held-out eval is flat or falling "
                f"({eval_slope:+.4f}) -- the classic reward-hacking signature",
                **evidence,
            )
        return self.ok("no reward-hacking signature", **evidence)


class ReplDegeneracyProbe(BaseProbe):
    """Experiment-specific kill criterion, pre-registered rather than discovered.

    With sub-LM calls removed, the lazy strategy is `print(context[:4000])` and
    then answering from the truncated slice -- ignoring the REPL entirely while
    still scoring passably. If mean REPL turns and the non-trivial-operation
    fraction fall while reward rises, the run is no longer measuring context
    management, whatever the reward curve says.
    """

    name = "rlm.repl_degeneracy"

    def check(self, ctx: Context) -> Verdict:
        try:
            turns = read_window(ctx, "repl/mean_turns", 20)
            nontrivial = read_window(ctx, "repl/nontrivial_fraction", 20)
            reward = read_window(ctx, "reward/mean", 20)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read REPL metrics: {exc}")
        if len(turns) < 10 or len(reward) < 10:
            return self.unknown("not enough REPL history to judge degeneracy")

        turn_slope, reward_slope = trend(turns), trend(reward)
        nontrivial_slope = trend(nontrivial) if len(nontrivial) >= 10 else 0.0
        evidence = {
            "mean_turns": round(turns[-1], 2),
            "turn_slope": round(turn_slope, 5),
            "nontrivial_fraction": round(nontrivial[-1], 3) if nontrivial else None,
            "nontrivial_slope": round(nontrivial_slope, 5),
            "reward_slope": round(reward_slope, 5),
        }
        if reward_slope > 0 and turn_slope < 0 and nontrivial_slope <= 0:
            return self.fail(
                f"reward rising ({reward_slope:+.4f}/step) while REPL turns fall "
                f"({turn_slope:+.4f}/step) and non-trivial operations do not -- the "
                f"policy is scoring by truncating context, not by managing it. This is "
                f"the pre-registered kill criterion.",
                **evidence,
            )
        return self.ok(f"REPL usage healthy ({turns[-1]:.1f} mean turns)", **evidence)


# --- regime: sft --------------------------------------------------------------


class LossPlateauProbe(BaseProbe):
    """Loss with no movement at all over the plateau window."""

    name = "rlm.loss_plateau"

    def check(self, ctx: Context) -> Verdict:
        window = ctx.cfg.health.plateau_window
        try:
            values = read_window(ctx, "train/loss", window)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read train/loss: {exc}")
        if len(values) < window:
            return self.unknown(f"only {len(values)} of {window} steps of loss history")

        spread = statistics.pstdev(values)
        evidence = {"window": window, "stdev": round(spread, 6),
                    "first": round(values[0], 4), "last": round(values[-1], 4)}
        if spread < 1e-6:
            return self.warn(
                f"loss has not moved over {window} steps (stdev {spread:.2e})", **evidence
            )
        return self.ok(f"loss moving (stdev {spread:.4f} over {window} steps)", **evidence)


class ReasoningTokenAccuracyProbe(BaseProbe):
    """Reasoning-span accuracy flat while total loss falls.

    The model is learning formatting rather than reasoning: exactly the outcome
    an SFT run on reasoning traces is meant to avoid, and invisible in the loss
    curve alone.
    """

    name = "rlm.reasoning_token_accuracy"

    def check(self, ctx: Context) -> Verdict:
        try:
            accuracy = read_window(ctx, "train/reasoning_token_accuracy", 20)
            loss = read_window(ctx, "train/loss", 20)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read reasoning accuracy: {exc}")
        if len(accuracy) < 10 or len(loss) < 10:
            return self.unknown("not enough history to compare accuracy against loss")

        acc_slope, loss_slope = trend(accuracy), trend(loss)
        evidence = {"accuracy_slope": round(acc_slope, 6), "loss_slope": round(loss_slope, 6),
                    "latest_accuracy": round(accuracy[-1], 4)}
        if loss_slope < 0 and abs(acc_slope) < 1e-5:
            return self.warn(
                "loss is falling while reasoning-span accuracy is flat -- the model is "
                "learning formatting, not reasoning", **evidence,
            )
        return self.ok(f"reasoning accuracy {accuracy[-1]:.3f}", **evidence)


class SequenceLengthProbe(BaseProbe):
    """p95 sequence length approaching max_seq_len.

    Truncation is silent and it removes the end of the trace -- which is where
    the conclusion is. Warning before the limit is the only useful time.
    """

    name = "rlm.sequence_length"

    def check(self, ctx: Context) -> Verdict:
        max_len = ctx.local.get("max_seq_len")
        if not max_len:
            return self.unknown("max_seq_len not reported")
        try:
            values = read_window(ctx, "train/seq_len_p95", 20)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read seq_len_p95: {exc}")
        if not values:
            return self.unknown("no seq_len_p95 logged yet")

        latest = values[-1]
        pct = latest / max_len * 100.0
        limit = ctx.cfg.health.seq_len_p95_warn_pct
        evidence = {"p95": round(latest, 1), "max_seq_len": max_len, "pct": round(pct, 1)}
        if pct >= limit:
            return self.warn(
                f"p95 sequence length is {pct:.0f}% of max_seq_len -- truncation is "
                f"silently discarding the end of traces, where the conclusions are",
                **evidence,
            )
        return self.ok(f"p95 sequence length {pct:.0f}% of max", **evidence)


class PackingEfficiencyProbe(BaseProbe):
    """Falling packing efficiency = wasted compute per step."""

    name = "rlm.packing_efficiency"

    def check(self, ctx: Context) -> Verdict:
        try:
            values = read_window(ctx, "train/packing_efficiency", 20)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read packing efficiency: {exc}")
        if not values:
            return self.unknown("no packing efficiency logged yet")

        latest, slope = values[-1], trend(values)
        evidence = {"latest": round(latest, 3), "slope_per_step": round(slope, 5)}
        if latest < 0.6:
            return self.warn(
                f"packing efficiency {latest:.0%} -- a large fraction of each step is "
                f"padding", **evidence,
            )
        return self.ok(f"packing efficiency {latest:.0%}", **evidence)


class HeldOutEvalProbe(BaseProbe):
    """Non-monotone decline across evals: overfitting.

    Warns and suggests a halt rather than failing. Overfitting is a judgement
    call about diminishing returns, not a broken run, and this library does not
    make that call on its own.
    """

    name = "rlm.held_out_eval"

    def check(self, ctx: Context) -> Verdict:
        patience = ctx.cfg.health.eval_patience
        try:
            values = read_window(ctx, "eval/score", patience + 1)
        except WandbUnavailable as exc:
            return self.unknown(f"cannot read eval/score: {exc}")
        if len(values) < patience + 1:
            return self.unknown(f"only {len(values)} evals so far, need {patience + 1}")

        best = max(values)
        recent = values[-patience:]
        evidence = {"best": round(best, 4), "recent": [round(v, 4) for v in recent],
                    "patience": patience}
        if all(v < best for v in recent):
            return self.warn(
                f"held-out eval has been below its best ({best:.3f}) for {patience} "
                f"consecutive evals -- likely overfitting; consider halting",
                **evidence,
            )
        return self.ok(f"held-out eval {values[-1]:.3f} (best {best:.3f})", **evidence)


RL_PROBES: tuple[BaseProbe, ...] = (
    RewardStdProbe(),
    KLDivergenceProbe(),
    EntropyProbe(),
    FormatSuccessProbe(),
    RolloutDurationProbe(),
    GenerationBackendProbe(),
    CompletionLengthProbe(),
    ClipFractionProbe(),
    VramHeadroomProbe(),
    RewardModelLatencyProbe(),
    RewardHackingProbe(),
    ReplDegeneracyProbe(),
)

SFT_PROBES: tuple[BaseProbe, ...] = (
    LossPlateauProbe(),
    ReasoningTokenAccuracyProbe(),
    SequenceLengthProbe(),
    PackingEfficiencyProbe(),
    HeldOutEvalProbe(),
)


def probes_for_regime(regime: str) -> tuple[BaseProbe, ...]:
    """Select the probe set. `regime` is validated at config load."""
    return RL_PROBES if regime == "rl" else SFT_PROBES
