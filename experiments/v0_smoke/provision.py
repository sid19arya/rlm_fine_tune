#!/usr/bin/env python3
"""Provision the V0 smoke-test pod: 2x A40, 30GB container, 100GB volume.

Kept out of `rlmwatch` on purpose. The library's RunPod client is deliberately
narrow -- status, logs, stop, terminate -- because it is the only component that
can spend or stop spending money and it stays small enough to audit in one
sitting. Pod *creation* is experiment-specific and also spends money, so it
lives here where it is read alongside the run it provisions.

Two refusals are built in:

* `--spend-cap-confirmed` is required. Nothing can verify through the API that
  you set an account spending limit, so the flag is you asserting it. The
  account default is $80/hr, which is not a safety net.
* A network volume is never requested. It pins the datacenter, limits GPU
  availability, and bills $0.07/GB/month indefinitely. The adapter goes to HF
  before the pod is terminated instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

RUNPOD_API = "https://rest.runpod.io/v1"

# A40 first, A6000 as the documented fallback when A40 is out of stock.
GPU_PREFERENCES = (
    ("NVIDIA A40", 0.44),
    ("NVIDIA RTX A6000", 0.49),
)

POD_SPEC = {
    "cloudType": "SECURE",
    "gpuCount": 2,
    "containerDiskInGb": 30,
    "volumeInGb": 100,
    "volumeMountPath": "/workspace",
    "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "ports": ["22/tcp", "8000/http"],
    # No networkVolumeId. See the module docstring.
}


class ProvisionError(RuntimeError):
    pass


def _client(api_key: str) -> httpx.Client:
    return httpx.Client(
        base_url=RUNPOD_API,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=60.0,
    )


def available_gpu(client: httpx.Client) -> tuple[str, float]:
    """Pick the first preferred GPU type that is actually in stock."""
    response = client.get("/gpuTypes")
    if response.status_code in (401, 403):
        raise ProvisionError(
            "RunPod rejected the API key. It must exist and have write scope -- a "
            "read-only key can list GPUs but cannot create or terminate a pod, which "
            "means nothing could stop the spend once it starts."
        )
    response.raise_for_status()
    catalogue = {g.get("displayName") or g.get("id"): g for g in response.json()}

    for name, expected_price in GPU_PREFERENCES:
        entry = catalogue.get(name)
        if entry is None:
            continue
        stock = entry.get("secureCloud", entry.get("stockStatus"))
        if stock in (False, "None", "none", 0):
            print(f"  {name}: out of stock, trying the next preference")
            continue
        price = float(entry.get("securePrice") or entry.get("costPerHr") or expected_price)
        return str(entry.get("id") or name), price

    raise ProvisionError(
        f"none of {[n for n, _ in GPU_PREFERENCES]} are available in Secure Cloud. "
        f"Wait, or escalate -- do not silently substitute a different GPU, because "
        f"the seconds-per-step number this run exists to produce would not transfer."
    )


def create_pod(client: httpx.Client, gpu_id: str, name: str, env: dict[str, str]) -> dict:
    payload = {
        **POD_SPEC,
        "name": name,
        "gpuTypeIds": [gpu_id],
        "env": env,
    }
    response = client.post("/pods", json=payload)
    if response.status_code >= 400:
        raise ProvisionError(f"pod creation failed ({response.status_code}): {response.text}")
    return response.json()


def wait_until_running(client: httpx.Client, pod_id: str, timeout_s: float = 600.0) -> dict:
    """Poll until the pod is RUNNING, or give up and say so.

    A pod stuck in PENDING is still being billed from provisioning, so the
    timeout here is a cost control, not a convenience.
    """
    deadline = time.monotonic() + timeout_s
    last = {}
    while time.monotonic() < deadline:
        response = client.get(f"/pods/{pod_id}")
        if response.status_code < 400:
            last = response.json()
            state = str(last.get("desiredStatus") or last.get("status") or "").upper()
            if state == "RUNNING":
                return last
            print(f"  pod {pod_id}: {state or 'unknown'}...")
        time.sleep(10)
    raise ProvisionError(
        f"pod {pod_id} did not reach RUNNING within {timeout_s:.0f}s. It is still "
        f"billing from the moment it was provisioned. Check the console, and "
        f"terminate it if it is wedged:\n"
        f"    rlmwatch kill -c configs/rlm-ft-v0-smoke.yaml --pod-id {pod_id} "
        f"--action terminate --yes"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Provision the V0 smoke pod (2x A40, no network volume).",
    )
    parser.add_argument(
        "--spend-cap-confirmed",
        action="store_true",
        help="assert that an account spending limit is set. Required: the API "
             "cannot verify it, and the default cap of $80/hr is not a safety net.",
    )
    parser.add_argument("--name", default="rlm-v0-smoke")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the exact request and exit without spending")
    args = parser.parse_args(argv)

    if not args.spend_cap_confirmed:
        print(
            "Refusing to provision.\n\n"
            "Set a spending limit first: RunPod console -> Billing -> spending limit,\n"
            "and enable low-balance notifications. A forgotten 2x A40 pod is ~$21/day,\n"
            "and the account default cap of $80/hr will not stop that.\n\n"
            "Then re-run with --spend-cap-confirmed.",
            file=sys.stderr,
        )
        return 2

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("RUNPOD_API_KEY is not set.", file=sys.stderr)
        return 2

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print(
            "HF_TOKEN is not set. It is needed to pull Qwen3-8B and, more importantly, "
            "to push the adapter before the pod's volume disk is destroyed. Set it now "
            "rather than discovering it at the end of a successful run.",
            file=sys.stderr,
        )
        return 2

    env = {
        "HF_HOME": "/workspace/hf",  # keep weights OFF the 30GB container disk
        "HF_TOKEN": hf_token,
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "RUNPOD_API_KEY": api_key,  # the in-pod watchdog needs to be able to terminate
    }

    if args.dry_run:
        redacted = {k: ("<set>" if v else "") for k, v in env.items()}
        print(json.dumps({**POD_SPEC, "name": args.name, "env": redacted}, indent=2))
        print("\ndry run: nothing was created and nothing is billing.")
        return 0

    with _client(api_key) as client:
        print("checking GPU availability...")
        gpu_id, price = available_gpu(client)
        hourly = price * POD_SPEC["gpuCount"]
        print(f"  using {gpu_id} x{POD_SPEC['gpuCount']} at ${price:.2f}/hr each "
              f"= ${hourly:.2f}/hr")
        print(f"  projected 3h smoke test: ${hourly * 3:.2f}")

        pod = create_pod(client, gpu_id, args.name, env)
        pod_id = pod.get("id")
        if not pod_id:
            raise ProvisionError(f"pod created but no id returned: {pod}")
        print(f"\npod {pod_id} created. Billing started at provisioning, not at launch.")

        pod = wait_until_running(client, pod_id)

    print("\n--- next steps ---")
    print(f"export RUNPOD_POD_ID={pod_id}")
    print(f"  hourly rate:  ${hourly:.2f}/hr  (put this in the monitor config's "
          f"budget.hourly_rate_usd -- measure it, do not guess)")
    print("  ssh:          check the console for the mapped port, then")
    print("                ssh root@<ip> -p <port>")
    print("  on the pod:   bash setup.sh   (nvidia-smi first: expect 2 GPUs @ 48GB)")
    print("\nIf anything goes wrong from here, the pod is billing until you stop it:")
    print(f"    rlmwatch kill -c configs/rlm-ft-v0-smoke.yaml --pod-id {pod_id} "
          f"--action terminate --yes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProvisionError as exc:
        print(f"\nPROVISION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
