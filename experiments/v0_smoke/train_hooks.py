#!/usr/bin/env python3
"""Worked example: attaching rlmwatch to the V0 training loop.

This is the shape the spec asks for -- gate, watchdog, shutdown wrapper, and
nothing else in the training script. Everything tunable is in
`configs/rlm-ft-v0-smoke.yaml`.

`rl_hooks()` below returns the `StartupHooks` for this specific run. Wire the
four callables into whatever prime-rl exposes and the gate becomes real rather
than mostly-unwired; leave one out and the gate reports it as unwired rather
than quietly counting it as a pass.

Run this file directly to see the loop structure with a stub trainer:

    python train_hooks.py --demo

The demo runs the whole supervision path against a fake trainer, with the
startup gate skipped and no RunPod client, so the integration can be checked
without a pod and without a credential that could stop one.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time

import rlmwatch
from rlmwatch.probes.startup import StartupHooks

CONFIG = "configs/rlm-ft-v0-smoke.yaml"
INFERENCE_URL = os.environ.get("RLM_INFERENCE_URL", "http://localhost:8000")


# --- the four hooks the gate wants -------------------------------------------


def rl_hooks(trainer: object | None = None) -> StartupHooks:
    """Lend the startup gate enough to prove the run can actually work.

    Each of these is cheap relative to the six hours it can save, and each maps
    to a failure that otherwise surfaces late:

    * `load_model` -- wrong checkpoint, wrong parameter count
    * `first_batch` -- a sample longer than max_seq_len, truncated silently
    * `train_step` -- the warm-up step, which also fixes the throughput baseline
    * `rollout_cycle` -- the measurement `stall_threshold: auto` needs, so a
      long GRPO rollout is judged against 4x its real duration rather than a
      guessed floor
    """

    def load_model() -> dict:
        # Replace with the real load. The expected count catches the case where
        # a config typo silently loads a different model than the one billed for.
        raise NotImplementedError(
            "wire this to prime-rl's model load and return "
            "{'param_count': n, 'expected_param_count': 8_000_000_000}"
        )

    def first_batch() -> dict:
        raise NotImplementedError(
            "wire this to the dataloader and return "
            "{'max_seq_len': 2048, 'longest_sample': <tokens in the longest sample>}"
        )

    def train_step() -> float:
        raise NotImplementedError("wire this to one optimizer step; return the loss")

    def rollout_cycle() -> None:
        raise NotImplementedError("wire this to one rollout -> score -> update cycle")

    return StartupHooks(
        load_model=load_model,
        first_batch=first_batch,
        train_step=train_step,
        rollout_cycle=rollout_cycle,
    )


def generation_backend_healthy() -> bool:
    """vLLM's health endpoint. Only reachable from inside the pod.

    Without this the trainer blocks on a dead backend indefinitely and it shows
    up hours later as an unexplained stall.
    """
    import httpx

    try:
        return httpx.get(f"{INFERENCE_URL}/health", timeout=5.0).status_code < 400
    except Exception:  # noqa: BLE001 - unreachable is what we are reporting
        return False


class CheckpointHooks:
    """The cooperative side of L3: save at the next safe point, then halt.

    Without this the ladder cannot checkpoint and escalates straight from
    "something is wrong" to "terminate", losing whatever the run had learned.
    """

    def __init__(self) -> None:
        self.requested = False
        self.saved = False

    def request_checkpoint(self) -> None:
        # Must return promptly. The training loop notices the flag at its next
        # safe point; blocking here would stall the very loop being saved.
        self.requested = True

    def wait_for_checkpoint(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.saved:
                return True
            time.sleep(1.0)
        return False


# --- the loop ----------------------------------------------------------------


def train(*, demo: bool = False, max_steps: int = 20) -> int:
    """The three lines, in context."""
    checkpoints = CheckpointHooks()

    with rlmwatch.attach(
        CONFIG,
        hooks=None if demo else rl_hooks(),
        trainer=checkpoints,
        gen_backend_health=None if demo else generation_backend_healthy,
        skip_gate=demo,
        dry_run=demo,
    ) as watch:
        for step in range(max_steps):
            watch.phase("rollout")
            rollouts = _generate(demo)
            for _ in rollouts:
                # Inside a long phase, so a legitimately slow rollout stays
                # distinguishable from a wedged one.
                watch.heartbeat()

            watch.phase("update")
            _update(demo)

            watch.step(step + 1, max_steps=max_steps)

            if checkpoints.requested and not checkpoints.saved:
                watch.phase("checkpoint")
                _save_checkpoint(demo)
                checkpoints.saved = True

            if watch.halted:
                # The ladder decided the run should stop. Breaking here saves at
                # a safe point rather than wherever a signal happened to land.
                print(f"halt requested at step {step + 1}; stopping cleanly")
                break

    return 0


def _generate(demo: bool) -> list[int]:
    if demo:
        time.sleep(0.05)
        return list(range(4))
    raise NotImplementedError("wire to the orchestrator's rollout call")


def _update(demo: bool) -> float:
    if demo:
        time.sleep(0.02)
        return random.random()
    raise NotImplementedError("wire to the trainer's update call")


def _save_checkpoint(demo: bool) -> None:
    if demo:
        return
    raise NotImplementedError("wire to the trainer's checkpoint save")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--demo", action="store_true",
                        help="run the supervision path against a stub trainer, gate "
                             "skipped, no pod required")
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s: %(message)s")

    if not args.demo:
        print(
            "This file is a worked example, not a runnable trainer. Wire rl_hooks()\n"
            "and the three _generate/_update/_save_checkpoint stubs into prime-rl,\n"
            "or run --demo to exercise the supervision path with a stub.",
        )
        return 2

    return train(demo=True, max_steps=args.max_steps)


if __name__ == "__main__":
    raise SystemExit(main())
