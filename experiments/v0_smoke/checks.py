#!/usr/bin/env python3
"""The four V0 startup checks, at T+2 / T+5 / T+15 / T+30 minutes.

    python checks.py --at 2      # inference alive
    python checks.py --at 5      # read one trajectory BY EYE
    python checks.py --at 15     # first optimizer step
    python checks.py --at 30     # weight sync

Run them in order. **Stop at the first failure.** They are ordered by what they
rule out: there is no point reading a trajectory from an inference server that
never came up.

Two of these deserve their reputation.

**T+5 is the check people skip, and the one that matters.** It cannot pass or
fail on its own -- it prints a real trajectory and asks you five questions about
it. Everything automated downstream is checking that numbers move; this is the
only check that verifies the numbers mean what you think.

**T+30 is the classic silent failure.** The trainer updates, the inference
server never receives the new weights, and you sample from the base model for
the entire run while the loss curve looks perfect. There is no error anywhere.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOG_GLOB = "/workspace/logs/rlm_*.jsonl"
DEFAULT_INFERENCE_URL = "http://localhost:8000"

PASS, FAIL, EYES = "PASS", "FAIL", "NEEDS-EYES"


@dataclass
class Result:
    check: str
    status: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.status}] {self.check}: {self.detail}"


def _latest_log(pattern: str) -> Path | None:
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def _read_jsonl(path: Path, limit: int = 200) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= limit:
                break
    return rows


# --- T+2 ---------------------------------------------------------------------


def check_inference_alive(url: str) -> Result:
    """The model id comes back from the inference server."""
    import httpx

    try:
        response = httpx.get(f"{url}/v1/models", timeout=10.0)
    except Exception as exc:  # noqa: BLE001
        return Result(
            "T+2 inference alive", FAIL,
            f"{url}/v1/models is unreachable: {exc}. The trainer will wait on this "
            f"server indefinitely with no error, so nothing downstream will happen.",
        )
    if response.status_code >= 400:
        return Result("T+2 inference alive", FAIL,
                      f"{url}/v1/models returned {response.status_code}")

    try:
        ids = [m.get("id") for m in response.json().get("data", [])]
    except Exception:  # noqa: BLE001
        return Result("T+2 inference alive", FAIL, "response was not the expected JSON")
    if not ids:
        return Result("T+2 inference alive", FAIL, "server is up but serving no models")
    return Result("T+2 inference alive", PASS, f"serving {', '.join(str(i) for i in ids)}")


# --- T+5 ---------------------------------------------------------------------


def check_trajectory(log_glob: str) -> Result:
    """Print one trajectory and ask the five questions. Cannot self-certify."""
    path = _latest_log(log_glob)
    if path is None:
        return Result("T+5 trajectory", FAIL,
                      f"no trajectory logs matching {log_glob}; nothing has been rolled out")

    rows = _read_jsonl(path, limit=5)
    if not rows:
        return Result("T+5 trajectory", FAIL, f"{path} exists but is empty")

    iterations = rows[0].get("iterations") or []
    if not iterations:
        return Result("T+5 trajectory", FAIL,
                      f"{path}: first trajectory has no iterations -- the REPL loop never ran")

    print(f"\n--- first trajectory from {path} ---")
    for index, iteration in enumerate(iterations[:4]):
        code = str(iteration.get("code", "")).strip()
        stdout = str(iteration.get("stdout", "")).strip()
        print(f"\n[iteration {index}] code:")
        print("  " + "\n  ".join(code.splitlines()[:20] or ["<empty>"]))
        print(f"[iteration {index}] stdout:")
        print("  " + "\n  ".join(stdout.splitlines()[:12] or ["<empty>"]))

    # Automatable parts of the check, reported as evidence rather than a verdict.
    joined = json.dumps(rows[0])
    notes = []
    if "llm_query" in joined:
        notes.append("!! 'llm_query' appears in the trajectory -- sub-LM calls are NOT "
                     "stripped, and this run is answering a different question")
    if "Traceback" in joined:
        notes.append("!! a Python traceback appears in the trajectory output")
    ready = rows[0].get("answer", {}).get("ready")
    if ready is not None:
        notes.append(f"answer['ready'] on the first trajectory: {ready}")

    print("\n--- read the above and answer these five ---")
    print("  1. Did the ```repl block parse into runnable code?")
    print("  2. Did the code run without a traceback?")
    print("  3. Was `context` in scope, at roughly the expected length?")
    print("  4. Is there NO mention of llm_query anywhere?")
    print("  5. Did the loop end via answer['ready'], not by exhausting iterations?")
    for note in notes:
        print(f"\n  {note}")

    status = FAIL if any(n.startswith("!!") for n in notes) else EYES
    detail = ("automated red flags found -- see above" if status == FAIL else
              "printed for human reading; this check does not self-certify")
    return Result("T+5 trajectory", status, detail)


# --- T+15 --------------------------------------------------------------------


def check_first_optimizer_step(run_path: str | None) -> Result:
    """Loss present, grad norm nonzero, and reward std > 0.

    Zero reward variance means no gradient signal exists at all. The run will
    continue looking healthy indefinitely and learn nothing.
    """
    if not run_path:
        return Result("T+15 first step", FAIL,
                      "no W&B run path given (--wandb entity/project/run-id)")
    try:
        from rlmwatch.clients.wandb import WandbClient, WandbUnavailable
    except ImportError:
        return Result("T+15 first step", FAIL, "rlmwatch is not installed on this pod")

    client = WandbClient()
    try:
        loss = client.metric_window(run_path, "train/loss", 10)
        grad = client.metric_window(run_path, "train/grad_norm", 10)
        reward_std = client.metric_window(run_path, "reward/std", 10)
    except WandbUnavailable as exc:
        return Result("T+15 first step", FAIL, f"cannot read the run: {exc}")

    if not loss:
        return Result("T+15 first step", FAIL,
                      "no train/loss logged yet -- no optimizer step has completed")
    if not grad or grad[-1] == 0:
        return Result("T+15 first step", FAIL,
                      f"grad_norm is {grad[-1] if grad else 'absent'} -- no gradient is "
                      f"flowing, so the optimizer step was a no-op")
    if not reward_std:
        return Result("T+15 first step", FAIL, "no reward/std logged yet")
    if max(reward_std) <= 0:
        return Result(
            "T+15 first step", FAIL,
            f"reward std is {reward_std[-1]} -- every rollout in the group scored "
            f"identically, so the advantage is zero and there is no gradient signal "
            f"at all. Kill it; more steps cannot help.",
        )
    return Result("T+15 first step", PASS,
                  f"loss {loss[-1]:.4f}, grad_norm {grad[-1]:.3f}, "
                  f"reward std {reward_std[-1]:.4f}")


# --- T+30 --------------------------------------------------------------------


def check_weight_sync(log_glob: str, run_path: str | None) -> Result:
    """Did the inference server actually receive updated weights?

    The classic silent failure: the trainer updates, the broadcast never lands,
    and you sample from the base model for the whole run while the loss curve
    looks perfect. Nothing errors.
    """
    sync_markers = ("broadcast", "weight sync", "weights synced", "update_weights",
                    "load_weights", "sync_weights")

    found = []
    for path in sorted(glob.glob("/workspace/logs/*.log")) + sorted(glob.glob(log_glob)):
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for marker in sync_markers:
            if marker.lower() in text.lower():
                found.append(f"{Path(path).name}:{marker}")
                break

    if found:
        return Result("T+30 weight sync", PASS,
                      f"broadcast log line present ({found[0]})")

    return Result(
        "T+30 weight sync", FAIL,
        "no weight-broadcast log line found in any log. This is the failure where "
        "the trainer updates, the inference server never receives new weights, and "
        "you sample from the base model forever while the loss curve looks perfect. "
        "Confirm by hand before trusting anything: sample the SAME prompt against "
        f"{DEFAULT_INFERENCE_URL} now and again at step 15, and check the outputs "
        "actually diverged. If they are identical, the run is worthless -- kill it.",
    )


def _tmux_panes() -> Result:
    """Three panes: trainer, orchestrator, inference. Never a bare SSH shell."""
    try:
        out = subprocess.run(  # noqa: S603
            ["tmux", "list-panes", "-s", "-t", "rlm"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return Result("tmux session", FAIL, "tmux is not available")
    panes = [line for line in out.splitlines() if line.strip()]
    if not panes:
        return Result(
            "tmux session", FAIL,
            "no tmux session named 'rlm'. A dropped SSH connection kills an unwrapped "
            "run, and the pod keeps billing afterwards.",
        )
    return Result("tmux session", PASS, f"{len(panes)} pane(s) in session 'rlm'")


CHECKS = {2: "inference alive", 5: "read a trajectory", 15: "first optimizer step",
          30: "weight sync"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V0 startup checks.")
    parser.add_argument("--at", type=int, choices=sorted(CHECKS), required=True,
                        help="which check to run (minutes after launch)")
    parser.add_argument("--url", default=DEFAULT_INFERENCE_URL)
    parser.add_argument("--logs", default=DEFAULT_LOG_GLOB)
    parser.add_argument("--wandb", default=None, help="entity/project/run-id")
    args = parser.parse_args(argv)

    if args.at == 2:
        results = [_tmux_panes(), check_inference_alive(args.url)]
    elif args.at == 5:
        results = [check_trajectory(args.logs)]
    elif args.at == 15:
        results = [check_first_optimizer_step(args.wandb)]
    else:
        results = [check_weight_sync(args.logs, args.wandb)]

    print()
    for result in results:
        print(result)

    if any(r.status == FAIL for r in results):
        print("\nSTOP. Do not run the next check. Diagnose this one first, and if the "
              "same command fails twice, escalate rather than trying a third time.",
              file=sys.stderr)
        return 1
    if any(r.status == EYES for r in results):
        print("\nThis check does not self-certify. Read the trajectory above and answer "
              "the five questions before moving on.")
        return 0
    print(f"\nT+{args.at} passed. Next: "
          + (f"--at {min(k for k in CHECKS if k > args.at)}"
             if args.at < max(CHECKS) else "run to 20 steps and record seconds/step."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
