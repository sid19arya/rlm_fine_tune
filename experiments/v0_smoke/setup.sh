#!/usr/bin/env bash
# V0 smoke-test pod setup. Run this ON the pod, once, after SSHing in.
#
# ~25 minutes, almost all of it the Qwen3-8B download. The download is done
# here on purpose rather than lazily at step 1: a 16GB pull racing the first
# training step turns a config problem into a timeout that looks like a hang.
#
# set -u catches unset variables (a missing HF_TOKEN would otherwise fail
# silently at the very end, after the run has already succeeded); pipefail
# stops a failure mid-pipeline from being masked by a successful tail.
set -euo pipefail

log() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nSETUP FAILED: %s\n' "$*" >&2; exit 1; }

log "importing the pod's environment into this shell"
# Env vars passed at pod creation land in PID 1's environment, but RunPod only
# writes its OWN vars to /etc/rp_environment -- so an SSH session (and every
# tmux pane, which is another fresh shell) sees none of them. Left unfixed the
# trainer would find no WANDB_ENTITY/PROJECT/RUN_ID and W&B would invent a
# random run id in the default entity, which is exactly the silent failure the
# pinned run identity exists to prevent.
POD_ENV=/etc/rp_pod_env
# Values are shell-quoted with %q, not just prefixed with `export`. PUBLIC_KEY
# is "ssh-ed25519 AAAA... comment" -- three space-separated words -- so a naive
# `export PUBLIC_KEY=$value` splits and dies with
#   export: `rlm-v0-smoke': not a valid identifier
# PUBLIC_KEY is excluded outright: it is consumed by RunPod's start script to
# write authorized_keys and nothing downstream needs it in the environment.
POD_ENV_TMP=$(mktemp)
tr '\0' '\n' < /proc/1/environ | grep -E '^(WANDB_|HF_)' | while IFS='=' read -r k v; do
  printf 'export %s=%q\n' "$k" "$v"
done > "$POD_ENV_TMP"
mv "$POD_ENV_TMP" "$POD_ENV"
chmod 600 "$POD_ENV"
# shellcheck disable=SC1090
set -a; source "$POD_ENV"; set +a
grep -q "source $POD_ENV" ~/.bashrc || echo "source $POD_ENV" >> ~/.bashrc
echo "  imported: $(sed -E 's/=.*//; s/export //' "$POD_ENV" | tr '\n' ' ')"

: "${WANDB_RUN_ID:?WANDB_RUN_ID did not survive into this shell -- the monitor and \
Hermes address the run by id, and without it W&B will generate a random one.}"

log "verifying the hardware matches what was provisioned"
nvidia-smi || die "nvidia-smi is not available; this is not a GPU pod"

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[ "$gpu_count" -eq 2 ] || die "expected 2 GPUs, found ${gpu_count}. \
prime-rl needs the trainer and the inference server on the same node, so a \
1-GPU pod cannot run this at all."

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
[ "$free_mb" -gt 40000 ] || die "only ${free_mb}MB free on the emptiest GPU. \
Something is already resident; this will OOM later rather than now."

log "verifying the volume disk"
df -h /workspace || die "/workspace is not mounted"
free_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge 80 ] || die "only ${free_gb}G free on /workspace, expected ~100G"

# The container disk is 30GB. Qwen3-8B alone is ~16GB, and HF's default cache
# is on the container disk. Every "no space left on device" three hours in
# traces back to skipping this line.
log "pinning the HF and uv caches to the volume disk"
export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME"
grep -q 'HF_HOME=/workspace/hf' ~/.bashrc || echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc

# uv's cache defaults to ~/.cache/uv, which is on the 30GB container disk.
# prime-rl pulls torch, vLLM and ~300 other packages; the cache alone reached
# 15GB of a 30GB disk on this pod, with the OS already using 13GB. Left on the
# container disk this is a "no space left on device" three hours in -- the same
# failure HF_HOME is pinned to avoid, via a different cache.
export UV_CACHE_DIR=/workspace/uv-cache
mkdir -p "$UV_CACHE_DIR"
grep -q 'UV_CACHE_DIR=/workspace/uv-cache' ~/.bashrc   || echo 'export UV_CACHE_DIR=/workspace/uv-cache' >> ~/.bashrc
# Same filesystem as the venvs, so uv can hardlink instead of full-copying --
# which also removes the "Failed to hardlink files" warning and the doubled
# disk write it implies.
export UV_LINK_MODE=hardlink

: "${HF_TOKEN:?HF_TOKEN is not set. It is needed to push the adapter before the \
volume disk is destroyed on terminate. Set it now, not at the end.}"

log "installing uv"
cd /workspace
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

log "cloning and installing the rlm fork, pinned"
# Our fork, not upstream. It carries the enable_sub_lm flag that selects the
# no-recursion arm. Pinned to an exact commit so the harness version is part
# of the run's record: a seconds-per-step number is only comparable against
# the code that produced it.
RLM_REPO="${RLM_REPO:-https://github.com/sid19arya/rlm}"
RLM_SHA="${RLM_SHA:-edfe8548f53c265cdc40ac8cf2784137d2707c28}"
[ -d /workspace/rlm ] || git clone "$RLM_REPO" /workspace/rlm
cd /workspace/rlm
git fetch -q origin && git checkout -q "$RLM_SHA"
echo "  rlm pinned at $(git rev-parse --short HEAD)"
# --allow-existing, not --clear: the script is meant to be re-runnable after
# a late failure, and --clear would throw away installs that already
# succeeded. Plain `uv venv` hard-fails on an existing .venv.
uv venv --python 3.12 --allow-existing
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e .

log "cloning prime-rl"
[ -d /workspace/prime-rl ] || git clone https://github.com/PrimeIntellect-ai/prime-rl \
  /workspace/prime-rl
cat <<'NOTE'

prime-rl installs per its own README, which changes more often than this script
does. Follow it now, in this same venv, then re-run this script -- it is
idempotent and will skip everything already done.

NOTE

log "installing the monitor and its extras"
# This repo has to be ON the pod, not just referenced. Cloning is the
# reproducible option; if you are iterating on rlmwatch locally, rsync your
# working copy to /workspace/rlm_fine_tune before running this and the clone
# is skipped.
WATCH_REPO="${WATCH_REPO:-https://github.com/sid19arya/rlm_fine_tune}"
[ -d /workspace/rlm_fine_tune ] || git clone "$WATCH_REPO" /workspace/rlm_fine_tune
uv pip install -e /workspace/rlm_fine_tune
uv pip install "wandb>=0.17" py-spy

log "installing the smoke config into the rlm checkout"
# rlm reads its own config directory, so the overlay config is copied in rather
# than referenced. Copied on every run so an edit in this repo cannot be
# silently shadowed by a stale copy from a previous setup.
mkdir -p /workspace/rlm/training/configs
cp /workspace/rlm_fine_tune/experiments/v0_smoke/smoke.toml \
   /workspace/rlm/training/configs/smoke.toml
echo "  training/configs/smoke.toml written"

log "installing tmux (not in the base image; the old instructions assumed it)"
# The previous next-steps block told the operator to run under tmux on an image
# that has no tmux, so the recommended command failed outright. launch.sh no
# longer needs it, but an interactive shell for poking at a live run does.
apt-get install -y -qq tmux < /dev/null > /dev/null 2>&1   && echo "  tmux $(tmux -V 2>/dev/null)" || echo "  tmux unavailable (non-fatal)"

log "pre-downloading Qwen3-8B"
# Not during step 1. A 16GB pull racing the first training step produces a
# timeout that looks exactly like a hang.
#
# Via the Python API rather than the CLI. `huggingface-cli` is not installed by
# any of rlm's dependencies, and in recent huggingface_hub versions the command
# was renamed to `hf` anyway -- so calling either by name is a coin flip that
# fails with `command not found` after everything else has already succeeded.
# snapshot_download is the same code path and is stable across versions.
uv pip install -q "huggingface_hub>=0.24"
python - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    "Qwen/Qwen3-8B",
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=8,
)
total = sum(f.stat().st_size for f in __import__("pathlib").Path(path).rglob("*")
            if f.is_file())
print(f"model at {path} ({total / 1024**3:.1f} GiB)")
PY

log "setup complete"
cat <<'NEXT'
Next:

  1. Confirm the ablation arm is actually selected:
       python /workspace/rlm_fine_tune/experiments/v0_smoke/verify_ablation.py
     (smoke.toml sets enable_sub_lm = false; this checks the REPL and the
      prompt agree, and that nothing re-binds the tools mid-rollout)
  2. rlmwatch preflight -c configs/rlm-ft-v0-smoke.yaml
  3. bash /root/launch.sh     <-- use this, NOT a bare `uv run rl`.
                                  It detaches (so an SSH drop cannot kill the
                                  run), records the trainer's exit code, and
                                  turns on faulthandler + unbuffered output.
                                  Launching by hand is how four startup deaths
                                  were made undiagnosable.

  Artifacts land in /workspace/runs/<timestamp>/:
    train.log  exit_code  dmesg.txt

Then the four checks, in order, stopping at the first failure:
  python checks.py --at 2 ; --at 5 ; --at 15 ; --at 30
NEXT
