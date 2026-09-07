#!/usr/bin/env python3
"""Upload the V0 artifacts to Hugging Face, then -- and only then -- terminate.

**The pod's volume disk is deleted on terminate, and there is no network volume
by design.** Everything worth keeping has to leave the pod first.

So the order here is not a preference, it is the whole point:

    1. measure seconds/step from the run's own metrics
    2. upload the adapter, the logs and the measurement to HF
    3. verify the upload by reading the repo back
    4. only if (3) succeeded, offer to terminate

Step 4 is refused outright if step 3 did not pass. A successful run whose
adapter was deleted is worse than a failed run, because it also costs the time
to do it again.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUTPUTS = "/workspace/outputs"
DEFAULT_LOGS = "/workspace/logs"


def find_adapter(outputs: str) -> Path | None:
    """The newest directory containing LoRA adapter weights."""
    candidates: list[Path] = []
    for pattern in ("**/adapter_model.safetensors", "**/adapter_model.bin",
                    "**/adapter_config.json"):
        candidates.extend(Path(p).parent for p in glob.glob(f"{outputs}/{pattern}",
                                                            recursive=True))
    if not candidates:
        return None
    return max(set(candidates), key=lambda p: p.stat().st_mtime)


def measure_seconds_per_step(run_path: str | None, *, max_steps: int = 20) -> dict:
    """The actual deliverable of V0.

    Everything else in the smoke test is plumbing verification. This number is
    what decides whether V1 is affordable, so it is computed from the run's own
    logged timestamps rather than from a stopwatch.
    """
    if not run_path:
        return {"error": "no W&B run path given; measure by hand from the step timestamps"}
    try:
        from rlmwatch.clients.wandb import WandbClient, WandbUnavailable
    except ImportError:
        return {"error": "rlmwatch not installed; cannot read the run"}

    client = WandbClient()
    try:
        timestamps = client.metric_window(run_path, "_timestamp", max_steps + 5)
        summary = client.summary(run_path)
    except WandbUnavailable as exc:
        return {"error": f"cannot read the run: {exc}"}

    if len(timestamps) < 2:
        return {"error": f"only {len(timestamps)} step timestamps; nothing to measure"}

    span = timestamps[-1] - timestamps[0]
    steps = len(timestamps) - 1
    seconds_per_step = span / steps

    v1_steps = 250
    return {
        "seconds_per_step": round(seconds_per_step, 1),
        "steps_measured": steps,
        "wall_clock_s": round(span, 1),
        "final_step": summary.get("_step"),
        # The reason the number exists: does V1 fit its budget?
        "v1_projection": {
            "steps": v1_steps,
            "hours": round(seconds_per_step * v1_steps / 3600, 1),
            "usd_at_2x_a100": round(seconds_per_step * v1_steps / 3600 * 2 * 1.39, 2),
            "budget_usd": 60,
        },
    }


def write_measurement(path: Path, measurement: dict, run_path: str | None) -> Path:
    payload = {
        "experiment": "v0-smoke",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "wandb_run": run_path,
        **measurement,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def upload(repo: str, adapter: Path, extras: list[Path], *, token: str) -> bool:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed: pip install huggingface_hub",
              file=sys.stderr)
        return False

    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True, private=True)

    print(f"uploading adapter from {adapter} ...")
    api.upload_folder(repo_id=repo, folder_path=str(adapter), path_in_repo="adapter")
    for extra in extras:
        if not extra.exists():
            continue
        target = f"logs/{extra.name}" if extra.is_file() else "logs"
        print(f"uploading {extra} -> {target}")
        if extra.is_file():
            api.upload_file(repo_id=repo, path_or_fileobj=str(extra), path_in_repo=target)
        else:
            api.upload_folder(repo_id=repo, folder_path=str(extra), path_in_repo=target)
    return True


def verify_upload(repo: str, *, token: str) -> bool:
    """Read the repo back. An upload that was not verified did not happen."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False
    try:
        files = HfApi(token=token).list_repo_files(repo_id=repo, repo_type="model")
    except Exception as exc:  # noqa: BLE001
        print(f"could not read {repo} back: {exc}", file=sys.stderr)
        return False
    adapter_files = [f for f in files if f.startswith("adapter/")]
    print(f"  {repo} now holds {len(files)} file(s), {len(adapter_files)} in adapter/")
    return bool(adapter_files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hf-repo", required=True, help="e.g. you/rlm-smoke-8b")
    parser.add_argument("--outputs", default=DEFAULT_OUTPUTS)
    parser.add_argument("--logs", default=DEFAULT_LOGS)
    parser.add_argument("--wandb", default=None, help="entity/project/run-id")
    parser.add_argument("--terminate", action="store_true",
                        help="terminate the pod after a VERIFIED upload")
    parser.add_argument("--pod-id", default=os.environ.get("RUNPOD_POD_ID", ""))
    args = parser.parse_args(argv)

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("HF_TOKEN is not set. Nothing can leave this pod without it, and the "
              "volume disk dies with the pod.", file=sys.stderr)
        return 2

    print("=== measuring seconds per step (the V0 deliverable) ===")
    measurement = measure_seconds_per_step(args.wandb)
    print(json.dumps(measurement, indent=2))

    projection = measurement.get("v1_projection")
    if projection and projection["usd_at_2x_a100"] > projection["budget_usd"]:
        print(
            f"\nNOTE: at {measurement['seconds_per_step']}s/step, V1's 250 steps project "
            f"to ${projection['usd_at_2x_a100']} against a ${projection['budget_usd']} "
            f"budget. That is a go/no-go input for the human, not a reason to change "
            f"the config.",
        )

    adapter = find_adapter(args.outputs)
    if adapter is None:
        print(f"\nNo LoRA adapter found under {args.outputs}. Nothing to upload, which "
              f"means exit criterion 4 was not met.", file=sys.stderr)
        return 1
    print(f"\nfound adapter: {adapter}")

    measurement_file = write_measurement(Path(args.outputs) / "v0_measurement.json",
                                         measurement, args.wandb)

    print("\n=== uploading ===")
    if not upload(args.hf_repo, adapter, [Path(args.logs), measurement_file], token=token):
        print("upload failed; the pod is NOT being terminated.", file=sys.stderr)
        return 1

    print("\n=== verifying ===")
    if not verify_upload(args.hf_repo, token=token):
        print(
            "\nUpload could not be verified. Refusing to terminate: the volume disk "
            "dies with the pod and there is no network volume. Re-upload, or copy the "
            "adapter off by hand, before stopping anything.",
            file=sys.stderr,
        )
        return 1

    print(f"\nartifacts are safe at https://huggingface.co/{args.hf_repo}")

    if not args.terminate:
        print(
            "\nThe pod is still RUNNING and still billing. Terminate when you are done "
            "reading the logs:\n"
            f"    python finish.py --hf-repo {args.hf_repo} --terminate\n"
            "  or\n"
            f"    rlmwatch kill -c configs/rlm-ft-v0-smoke.yaml --action terminate --yes"
        )
        return 0

    if not args.pod_id:
        print("--terminate needs --pod-id or RUNPOD_POD_ID.", file=sys.stderr)
        return 2

    from rlmwatch.clients.runpod import RunPodClient

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("RUNPOD_API_KEY is not set; cannot terminate.", file=sys.stderr)
        return 2

    print(f"\nterminating {args.pod_id} ...")
    RunPodClient(api_key).terminate(args.pod_id)
    print("terminated. Billing has stopped.")
    print("\nGATE: report the V0 results and wait for a human go/no-go before V1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
