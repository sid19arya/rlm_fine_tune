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
log "pinning the HF cache to the volume disk"
export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME"
grep -q 'HF_HOME=/workspace/hf' ~/.bashrc || echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc

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
uv venv --python 3.12
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

log "pre-downloading Qwen3-8B"
# Not during step 1. A 16GB pull racing the first training step produces a
# timeout that looks exactly like a hang.
huggingface-cli download Qwen/Qwen3-8B

log "setup complete"
cat <<'NEXT'
Next:

  1. Confirm the ablation arm is actually selected:
       python /workspace/rlm_fine_tune/experiments/v0_smoke/verify_ablation.py
     (smoke.toml sets enable_sub_lm = false; this checks the REPL and the
      prompt agree, and that nothing re-binds the tools mid-rollout)
  2. rlmwatch preflight -c configs/rlm-ft-v0-smoke.yaml
  3. tmux new -s rlm          <-- NEVER a bare SSH shell. A dropped
                                  connection kills an unwrapped run, and the
                                  pod keeps billing afterwards.
  5. uv run rl @ training/configs/smoke.toml

Then the four checks, in order, stopping at the first failure:
  python checks.py --at 2 ; --at 5 ; --at 15 ; --at 30
NEXT
