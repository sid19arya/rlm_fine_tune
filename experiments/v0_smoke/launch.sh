#!/usr/bin/env bash
# Launch the V0 training run ON the pod, instrumented so that a death is
# diagnosable. Run after setup.sh.
#
# WHY THIS FILE EXISTS
#
# The first four V0 launches were ad-hoc SSH one-liners of the shape
#   setsid nohup uv run rl --config smoke.toml > train.log 2>&1 &
# and every one of them died during startup "with no output". Three separate
# defects in that one line are responsible for how little was learned:
#
#   1. The exit status was discarded. `setsid ... &` detaches and nothing ever
#      reaped the child, so the single most informative byte -- 139 (SIGSEGV),
#      134 (SIGABRT), 137 (SIGKILL/OOM) vs. a plain 1 -- was thrown away four
#      times in a row. "Died silently" was a property of the launcher, not of
#      the trainer.
#   2. stdout was block-buffered. Redirected to a file, Python buffers stdout in
#      8KB blocks; a process killed by a signal never flushes. `train4.log` was
#      0 bytes, and that was read as "the process wrote nothing" -- but it only
#      ever meant "the process died before filling one 8KB buffer". Up to 8KB of
#      startup output existed and was destroyed at exit.
#   3. No fault handler. A segfault or CUDA abort in native code kills the
#      interpreter without a Python traceback, so the logs looked clean.
#
# Each is fixed below. The cost of the fix is a few environment variables; the
# cost of not fixing it was the whole diagnosis.
set -uo pipefail   # NOT -e: the whole point is to observe a failing command

RUN_DIR=${RUN_DIR:-/workspace/runs/$(date -u +%Y%m%dT%H%M%SZ)}
CONFIG=${CONFIG:-/workspace/rlm/training/configs/smoke.toml}
mkdir -p "$RUN_DIR"

LOG="$RUN_DIR/train.log"
EXIT_FILE="$RUN_DIR/exit_code"
DMESG_FILE="$RUN_DIR/dmesg.txt"

# shellcheck disable=SC1091
[ -f /etc/rp_pod_env ] && { set -a; source /etc/rp_pod_env; set +a; }

# --- instrumentation -------------------------------------------------------
# Dump a C-level stack on SIGSEGV/SIGBUS/SIGFPE/SIGABRT. This is the one that
# turns "native crash, cause unknown" into a named frame.
export PYTHONFAULTHANDLER=1
# Unbuffered stdout/stderr: output survives a signal death. Costs a little
# throughput on a chatty trainer and is worth it every time.
export PYTHONUNBUFFERED=1
# CUDA_LAUNCH_BLOCKING makes CUDA errors point at the launching call rather
# than a later, unrelated synchronisation point -- but it is OFF by default
# here, deliberately. It serialises every kernel launch, which changes timing:
# a race-condition crash can simply stop reproducing under it, turning a
# diagnosable bug into a Heisenbug and costing a pod-hour to learn nothing.
# It also slows startup, which is billed.
#
# Order of operations: run once with faulthandler + unbuffered only (below --
# near-zero perturbation), and set CUDA_LAUNCH_BLOCKING=1 for a SECOND pass
# only if the first stack lands inside CUDA/NCCL.
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
# Cores if the kernel will give them to us; harmless where it will not.
ulimit -c unlimited 2>/dev/null || true

echo "run dir : $RUN_DIR"
echo "config  : $CONFIG"
echo "faulthandler=$PYTHONFAULTHANDLER unbuffered=$PYTHONUNBUFFERED cuda_blocking=$CUDA_LAUNCH_BLOCKING"

# --- launch ----------------------------------------------------------------
# Foreground, inside a wrapper that records the status. stdbuf is belt-and-
# braces for any non-Python child that block-buffers on its own.
cd /workspace/prime-rl
stdbuf -o0 -e0 uv run rl --config "$CONFIG" > >(tee -a "$LOG") 2>&1
STATUS=$?

echo "$STATUS" > "$EXIT_FILE"
echo "=== trainer exited with status $STATUS ==="

# --- post-mortem -----------------------------------------------------------
# Signal deaths are reported by the shell as 128+N. Name the ones that matter
# so the number does not have to be looked up under pressure.
case "$STATUS" in
  0)   echo "clean exit" ;;
  134) echo "SIGABRT  -- assertion/abort in native code (CUDA, NCCL, glibc)" ;;
  137) echo "SIGKILL  -- almost always the OOM killer; check dmesg below" ;;
  139) echo "SIGSEGV  -- segfault; PYTHONFAULTHANDLER stack should be in the log" ;;
  *)   echo "exit $STATUS -- see $LOG" ;;
esac

# dmesg is frequently unreadable in an unprivileged container. Try anyway: when
# it works it settles OOM-vs-segfault in one line.
{ dmesg -T 2>/dev/null | tail -50; } > "$DMESG_FILE" 2>&1 || true
if [ -s "$DMESG_FILE" ]; then
  echo "--- dmesg tail ---"; cat "$DMESG_FILE"
else
  echo "(dmesg unavailable in this container)"
fi

echo "artifacts: $LOG  $EXIT_FILE  $DMESG_FILE"
exit "$STATUS"
