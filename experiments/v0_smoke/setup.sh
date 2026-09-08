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
# ...and persist these into the pod env file, not just ~/.bashrc. A detached
# launcher (setsid, nohup, cron) never sources ~/.bashrc, so it inherits
# UV_CACHE_DIR=unset and puts a ~15GB cache back on the 30GB container disk --
# the exact disk-exhaustion failure this pinning exists to prevent, arriving
# through a different door. Same class of bug as uv not being on PATH there.
for kv in "UV_CACHE_DIR=$UV_CACHE_DIR" "UV_LINK_MODE=$UV_LINK_MODE" "HF_HOME=$HF_HOME"; do
  grep -q "^export ${kv%%=*}=" /etc/rp_pod_env 2>/dev/null || echo "export $kv" >> /etc/rp_pod_env
done

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

# PIN prime-rl, do not float. HEAD renamed the env config key to `source`
# (d135ed4c9, 2026-07-29) and rejects rlm's config with 7 validation errors.
# 083127fe (2026-05-27) is the commit whose schema rlm's harness targets.
#
# How that was found, because two guesses from reading rename commit messages
# were wrong by weeks: rlm/training/configs/ has exactly one commit in its
# entire history (de762b9, 2026-05-24), which dates the harness and gives a
# three-day window. Date the consumer; do not read the producer's changelog.
#
# An earlier version of this script deferred the whole prime-rl install to its
# README, so this pin lived only in a shell history and was silently lost on
# the next pod -- which cost a full session.
PRIME_SHA="${PRIME_SHA:-083127fe}"
log "cloning prime-rl at pinned $PRIME_SHA"
[ -d /workspace/prime-rl ] || git clone https://github.com/PrimeIntellect-ai/prime-rl \
  /workspace/prime-rl
cd /workspace/prime-rl
git fetch --all --tags --quiet
git checkout --quiet "$PRIME_SHA" || die "could not check out prime-rl $PRIME_SHA"
echo "  prime-rl at $(git log --oneline -1)"

log "initialising prime-rl submodules"
# Two separate traps here.
#
# 1. Several submodules are declared with git@github.com: SSH URLs. The pod has
#    no GitHub key, so `submodule update` fails on them and leaves deps/ empty.
#    That surfaces much later, and much less clearly, as:
#      Failed to build `prime-pydantic-config` ... does not appear to be a
#      Python project, as neither pyproject.toml nor setup.py are present
#    These repos are all public over https.
#
# 2. At this pinned commit the submodule set includes configs/private ->
#    research-configs, which is a PRIVATE repo. git aborts the ENTIRE
#    `submodule update` when it cannot clone it ("Failed to clone
#    'configs/private' a second time, aborting") even though the four deps we
#    need cloned fine moments earlier. Scoping the update to `-- deps` skips it.
#    Nothing in this experiment reads configs/private.
sed -i 's#git@github.com:#https://github.com/#' .gitmodules
git submodule sync --quiet
git submodule update --init --recursive --depth 1 -- deps || die "submodule init failed"
for d in deps/*/; do
  [ -z "$(ls -A "$d" 2>/dev/null)" ] && die "submodule $d is empty after init"
done
echo "  deps ok: $(ls -d deps/*/ | tr '
' ' ')"

log "building prime-rl (torch, vLLM, ~300 packages -- the slow step)"
# --extra flash-attn is REQUIRED, not an optimisation. flash_attn is a
# transitive import of the trainer itself, not just of the attention backend
# this config selects:
#   rl.py -> trainer.model -> trainer.lora -> models/__init__ -> glm_moe_dsa
#        -> sparse_mla_attention -> utils.cp -> ring_flash_attn -> flash_attn
# so a plain `uv sync` builds an environment that cannot import the trainer at
# all, and fails with ModuleNotFoundError long after the config validates.
#
# It is cheap: the extra resolves to a PREBUILT wheel
# (flash_attn-2.8.3+cu128torch2.11-cp312-cp312-linux_x86_64.whl), not a source
# build, so this is a download rather than a 40-minute nvcc compile.
uv sync --extra flash-attn || die "uv sync failed"

log "wiring rlm into the prime-rl environment"
# Order matters: this runs AFTER uv sync. `uv sync` reconciles the environment
# against the lockfile and can evict packages installed with `uv pip install`,
# so wiring first and syncing second silently un-wires oolong.
uv pip install -e /workspace/rlm/training
uv pip install -e /workspace/rlm/training/environments/oolong
# orjson is a verifiers dependency that the lockfile carries but that has gone
# missing after re-resolution with --extra flash-attn. Cheap to assert.
uv pip install orjson
uv run python - <<'CHECK'
# Import the ACTUAL entrypoint chains, not just the top-level package.
#
# `import prime_rl` succeeds in an environment that cannot run anything: it
# touches none of the heavy submodules. Two separate missing dependencies got
# through a shallow check like that and only surfaced minutes into billed runs:
#
#   flash_attn  -- trainer.model -> trainer.lora -> models -> glm_moe_dsa
#                  -> sparse_mla_attention -> utils.cp -> ring_flash_attn
#   orjson      -- orchestrator -> advantage -> vf_utils
#
# The orchestrator one is the expensive kind: the run reports "Startup
# complete", brings up inference and the trainer on both GPUs, and only then
# dies -- about three minutes of GPU time to learn that an import is missing.
from importlib.metadata import entry_points

envs = [e.name for e in entry_points(group="verifiers.environments")]
assert "oolong" in envs, f"oolong not registered as a verifiers environment: {envs}"

import importlib
chains = [
    "prime_rl.trainer.model",            # pulls flash_attn transitively
    "prime_rl.orchestrator.orchestrator",  # pulls orjson transitively
    "prime_rl.entrypoints.rl",
    "rlm_train",
    "oolong",
]
failed = []
for mod in chains:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        failed.append(f"{mod}: {type(exc).__name__}: {exc}")
if failed:
    raise SystemExit("  entrypoint imports FAILED:
    " + "
    ".join(failed))

print(f"  verifiers environments: {envs}")
print(f"  entrypoint chains import OK ({len(chains)} checked)")
CHECK
cd - > /dev/null

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
