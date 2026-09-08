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
from pathlib import Path

import httpx

RUNPOD_API = "https://rest.runpod.io/v1"

# The W&B run's identity, in one place. Everything that has to name this run --
# the monitor config, the Hermes briefing, prime-rl's own [wandb] table -- has
# to agree with these, and a test asserts they do.
#
# WANDB_RUN_ID is set explicitly and is NOT the same thing as the display name
# in smoke.toml. W&B's API path is entity/project/<run-id>, and if the id is
# left to W&B it generates something random like "3xk9p2ab". The monitor and
# Hermes both need the path before the run exists, so the id is pinned here
# instead. Getting this wrong is silent: the path simply 404s and every probe
# reports `unknown` for the whole run.
WANDB_ENTITY = "rlm-runpod-1"
WANDB_PROJECT = "rlm-context-management"
DEFAULT_RUN_ID = "v0-smoke"

# `gpuTypeIds` takes RunPod's display names and is a list, which the API reads
# as "any of these". A40 first with the A6000 as the documented fallback, so
# stock-out is handled by RunPod at placement rather than by us polling a
# catalogue first. Both are Ampere and both were priced for this budget; nothing
# else belongs in this list, because a substituted GPU would make the
# seconds-per-step number this run exists to produce non-transferable.
GPU_PREFERENCES = ("NVIDIA A40", "NVIDIA RTX A6000")

#: Indicative only -- the real figure comes back on the created pod as
#: `costPerHr` and is what should go in the monitor config.
EXPECTED_PRICE_PER_GPU = {"NVIDIA A40": 0.44, "NVIDIA RTX A6000": 0.49}

POD_SPEC = {
    "computeType": "GPU",
    "cloudType": "SECURE",
    "gpuCount": 2,
    "containerDiskInGb": 30,
    # 200GB, not 100. Measured on a 100GB volume: the workload needs ~85-90GB
    # of real bytes and the volume charges quota at roughly 1.33x raw size, so
    # 100GB yields only ~75GB usable and the run dies mid-training with
    #   safetensors_rust.SafetensorError: Error while serializing:
    #   I/O error: Disk quota exceeded (os error 122)
    #
    # The budget, measured:
    #   uv environment (torch, vLLM, ~300 pkgs)   33GB
    #   Qwen3-8B in the HF cache                  20GB
    #   oolong-synth dataset                      11GB
    #   weight broadcast (8B bf16 via filesystem) ~16GB
    #
    # Do not trust statvfs here: `df` reports the MooseFS cluster (191T free)
    # rather than this volume's quota, so it shows enormous headroom on a
    # volume that is one write from EDQUOT. The only reliable check is to
    # write a probe file and see whether it lands.
    "volumeInGb": 200,
    "volumeMountPath": "/workspace",
    "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "ports": ["22/tcp", "8000/http"],
    "interruptible": False,   # a spot pod dying mid-run is not a saving here
    # No networkVolumeId, and no dataCenterIds: both pin placement, and pinning
    # is what makes a 2x A40 request fail on stock. See the module docstring.
}


class ProvisionError(RuntimeError):
    pass


def _client(api_key: str) -> httpx.Client:
    return httpx.Client(
        base_url=RUNPOD_API,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=60.0,
    )


def check_auth(client: httpx.Client) -> int:
    """Prove the key works and report how many pods already exist.

    Listing is RunPod's own connectivity test -- an empty list is a pass. It
    also catches the case where a previous run was left running: provisioning a
    second 2x A40 pod on top of a forgotten one doubles the burn silently.
    """
    response = client.get("/pods")
    if response.status_code in (401, 403):
        raise ProvisionError(
            "RunPod rejected the API key. It must exist and have write scope -- a "
            "read-only key can list pods but cannot create or terminate one, which "
            "means nothing could stop the spend once it starts."
        )
    response.raise_for_status()
    pods = response.json()
    return len(pods) if isinstance(pods, list) else 0


def create_pod(client: httpx.Client, name: str, env: dict[str, str]) -> dict:
    """Create the pod, letting RunPod pick from the preferred GPU list.

    `gpuTypeIds` is a list the API treats as "any of these", so stock-out
    falls back to the A6000 at placement time rather than needing a catalogue
    lookup first -- there is no v1 endpoint for that, and polling one would
    race the actual allocation anyway.
    """
    payload = {
        **POD_SPEC,
        "name": name,
        "gpuTypeIds": list(GPU_PREFERENCES),
        "env": env,
    }
    response = client.post("/pods", json=payload)
    if response.status_code >= 400:
        raise ProvisionError(
            f"pod creation failed ({response.status_code}): {response.text[:500]}\n"
            f"If this is a stock error, neither {' nor '.join(GPU_PREFERENCES)} has "
            f"2 GPUs free in Secure Cloud right now. Wait and retry rather than "
            f"substituting a different GPU."
        )
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
            # Readiness is "can I actually reach it", which means a public IP
            # and a mapped SSH port.
            #
            # `desiredStatus` is RUNNING from the instant the pod is rented --
            # it is the state RunPod is aiming for, not the state it is in, and
            # waiting on it returns in about four seconds with no IP.
            # `runtime` looks like the alternative but was still null on an
            # observed pod that was already accepting SSH, so waiting on that
            # would have timed out on a perfectly good pod.
            ip = last.get("publicIp") or ""
            ssh_port = (last.get("portMappings") or {}).get("22")
            if state == "RUNNING" and ip and ssh_port:
                return last
            if state != "RUNNING":
                detail = state or "unknown"
            elif not ip:
                detail = "booting (no IP yet)"
            else:
                detail = "booting (no SSH port yet)"
            print(f"  pod {pod_id}: {detail}...")
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
    parser.add_argument(
        "--run-id", default=DEFAULT_RUN_ID,
        help="W&B run id, which is what the monitor and Hermes address. Bump it "
             "on a retry (v0-smoke-2): WANDB_RESUME=never means a reused id fails "
             "rather than resuming into a half-finished run.",
    )
    parser.add_argument("--ssh-key", default="~/.ssh/runpod_rlm.pub",
                        help="public key injected as PUBLIC_KEY; RunPod writes it "
                             "to authorized_keys at container start")
    parser.add_argument("--allow-existing", action="store_true",
                        help="provision even though other pods are already billing")
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

    # Windows environment variables are case-insensitive; Linux ones are not.
    # A .env written as `wandb_api_key=` resolves fine here and then silently
    # does not exist on the pod, so the run trains with no W&B logging and the
    # whole monitor goes blind. Accept either spelling locally, always export
    # the canonical one.
    wandb_key = os.environ.get("WANDB_API_KEY") or os.environ.get("wandb_api_key", "")
    if not wandb_key:
        print(
            "WANDB_API_KEY is not set. The run would train with no metrics, which "
            "means every liveness and health probe reports `unknown` for the whole "
            "run and the sentinel is blind.",
            file=sys.stderr,
        )
        return 2

    # RunPod's PyTorch template writes PUBLIC_KEY into ~/.ssh/authorized_keys at
    # container start. It is injected at creation and cannot be added to a pod
    # that is already running, so a pod provisioned without it is unreachable
    # and has to be thrown away -- which is exactly what happened the first
    # time. Refuse up front rather than discover it after paying to boot.
    public_key = ""
    key_path = Path(args.ssh_key).expanduser()
    if key_path.is_file():
        public_key = key_path.read_text(encoding="utf-8").strip()
    if not public_key:
        print(
            f"No SSH public key at {key_path}. RunPod injects authorized_keys only at "
            f"pod creation, so a pod without one can never be logged into and has to "
            f"be terminated and rebuilt. Generate one first:\n"
            f"    ssh-keygen -t ed25519 -f {key_path.with_suffix('')} -N ''",
            file=sys.stderr,
        )
        return 2

    env = {
        "PUBLIC_KEY": public_key,
        "HF_HOME": "/workspace/hf",  # keep weights OFF the 30GB container disk
        "HF_TOKEN": hf_token,
        "WANDB_API_KEY": wandb_key,
        # Set as environment variables rather than trusting prime-rl's [wandb]
        # table, which has no `entity` key in the reference config. The W&B SDK
        # honours all of these regardless of what the trainer's config format
        # supports, so the run lands in a known place with a known id.
        "WANDB_ENTITY": WANDB_ENTITY,
        "WANDB_PROJECT": WANDB_PROJECT,
        "WANDB_RUN_ID": args.run_id,
        # Fail loudly if this id already exists rather than resuming into a
        # half-finished run -- resuming would splice two runs' step timings
        # together and corrupt the seconds-per-step measurement, which is the
        # only thing V0 produces.
        "WANDB_RESUME": "never",
        "RUNPOD_API_KEY": api_key,  # the in-pod watchdog needs to be able to terminate
    }

    if args.dry_run:
        # The exact payload create_pod would POST, with secrets masked. A dry
        # run that prints something other than the real request is worse than
        # no dry run at all.
        # Redact credentials only. The W&B identity and the cache path are not
        # secrets, and they are the part of this payload most worth eyeballing
        # before spending: a wrong entity or run id is invisible afterwards.
        secret_keys = {"HF_TOKEN", "WANDB_API_KEY", "RUNPOD_API_KEY"}
        redacted = {
            k: ("<set>" if v else "<MISSING>") if k in secret_keys
            else (v[:28] + "..." if k == "PUBLIC_KEY" and len(v) > 28 else v)
            for k, v in env.items()
        }
        payload = {
            **POD_SPEC,
            "name": args.name,
            "gpuTypeIds": list(GPU_PREFERENCES),
            "env": redacted,
        }
        print(json.dumps(payload, indent=2))
        print("\ndry run: nothing was created and nothing is billing.")
        return 0

    with _client(api_key) as client:
        print("checking auth and existing pods...")
        existing = check_auth(client)
        if existing:
            print(
                f"\nWARNING: {existing} pod(s) already exist on this account and are "
                f"billing. Provisioning another doubles the burn. Check the console "
                f"before continuing.",
                file=sys.stderr,
            )
            if not args.allow_existing:
                print("Refusing. Re-run with --allow-existing if that is intended.",
                      file=sys.stderr)
                return 2
        print(f"  auth ok, {existing} existing pod(s)")

        estimate = min(EXPECTED_PRICE_PER_GPU.values()) * POD_SPEC["gpuCount"]
        print(f"  requesting {POD_SPEC['gpuCount']}x from "
              f"{' or '.join(GPU_PREFERENCES)}, ~${estimate:.2f}/hr")

        pod = create_pod(client, args.name, env)
        pod_id = pod.get("id")
        if not pod_id:
            raise ProvisionError(f"pod created but no id returned: {pod}")
        print(f"\npod {pod_id} created. Billing started at provisioning, not at launch.")

        pod = wait_until_running(client, pod_id)

    # The real rate, from the pod itself. This is the figure that belongs in
    # the monitor config -- the list price is an estimate and can be wrong.
    hourly = float(pod.get("costPerHr") or 0.0)
    gpu = pod.get("machineType") or pod.get("gpuTypeId") or "?"
    public_ip = pod.get("publicIp") or "<see console>"
    # portMappings is a flat {"<privatePort>": <publicPort>} mapping, not a
    # list of objects. Getting this wrong crashed the success path of a
    # provision that had otherwise worked.
    ssh_port = str((pod.get("portMappings") or {}).get("22") or "<see console>")

    print("\n--- provisioned ---")
    print(f"  pod:          {pod_id}")
    print(f"  gpu:          {POD_SPEC['gpuCount']}x {gpu}")
    print(f"  measured rate ${hourly:.2f}/hr  -> 3h smoke test ~${hourly * 3:.2f}")
    print("\n--- next steps ---")
    print(f"  export RUNPOD_POD_ID={pod_id}")
    print(f"  set budget.hourly_rate_usd: {hourly:.2f}   (measured, not the list price)")
    print(f"  W&B run path: {WANDB_ENTITY}/{WANDB_PROJECT}/{args.run_id}")
    print(f"  ssh root@{public_ip} -p {ssh_port}")
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
