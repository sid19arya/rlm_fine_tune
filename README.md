# rlm_fine_tune

RL fine-tuning of a reasoning LM on RunPod GPUs, with the monitoring layer that
keeps an unattended run from quietly burning the budget.

| Directory | What it is |
|---|---|
| `rlmwatch/` | Reusable monitoring library — startup gate, in-pod watchdog, external sentinel, escalation ladder. Built and fault-tested *before* being wired to anything. |
| `experiments/v0_smoke/` | The V0 smoke test: 2× A40, 20 steps of RLM context-management RL, measure seconds/step, upload the adapter, terminate. |
| `configs/` | One validated YAML per run. No monitoring logic in the training script. |
| `tests/chaos/` | The nine injected faults the library has to survive before it is trusted. |

Specs live locally as `_instructions.md` (the experiment) and
`_monitoring_instructions.md` (the monitor); they are gitignored.
Both are normative — code follows them, not the other way round.

## The one-paragraph version

**Experiment.** With sub-LM calls stripped out, does RL improve a model's use of
a Python REPL over long context? V0 — this repo's current scope — only proves
the plumbing and produces one number: seconds per step. It *cannot* answer the
question: at 32 rollouts/step the per-step standard error is ±7.6 points, so a
real 5-point gain is invisible. V1 is not started until a human signs off on V0.

**Monitor.** RunPod tells you nothing when your script dies, hangs, or finishes
(the container restarts and the meter keeps running). W&B's heartbeat comes from
its own service process, so a deadlocked training loop shows `Running` forever.
`rlmwatch` closes that gap with two independent observers: a watchdog inside the
pod that can see stack traces but dies with the pod, and a sentinel outside it
that survives everything and holds the dead-man's switch.

## Quick start

```bash
cp .env.example .env          # RUNPOD_API_KEY (write scope), WANDB_API_KEY, HF_TOKEN
uv venv && uv pip install -e ".[dev]"
pytest                        # 298 tests, no GPU, no network, no cloud resources
```

Check the wiring without a pod:

```bash
RUNPOD_POD_ID=pod-demo python experiments/v0_smoke/train_hooks.py --demo
```

## Running V0

Full runbook in [`experiments/v0_smoke/README.md`](experiments/v0_smoke/README.md).
The short version:

```bash
# 0. Set a RunPod spending limit in the console. Nothing below checks this for
#    you, and the account default of $80/hr is not a safety net.
python experiments/v0_smoke/provision.py --spend-cap-confirmed
export RUNPOD_POD_ID=<printed pod id>

# on the pod
bash setup.sh
python strip_sub_lm_calls.py --apply && python strip_sub_lm_calls.py --verify
rlmwatch preflight -c configs/rlm-ft-v0-smoke.yaml
tmux new -s rlm && uv run rl @ training/configs/smoke.toml

# from your laptop, NOT the pod
rlmwatch watch -c configs/rlm-ft-v0-smoke.yaml

# gates: stop at the first failure
python checks.py --at 2 ; --at 5 ; --at 15 ; --at 30

# artifacts leave before the disk dies
python finish.py --hf-repo <you>/rlm-smoke-8b
```

## How the monitor is put together

```
   ┌──────────────────── RunPod pod ─────────────────────┐
   │  train.py                                            │
   │    with rlmwatch.attach(cfg) as watch:   ← 3 lines   │
   │        watch.phase(...) / heartbeat() / step(...)    │
   └──────────────────────────────────────────────────────┘
            │ wandb.log                    ▲ stop / terminate
            ▼                              │
     ┌─────────────┐              ┌────────────────┐
     │  W&B cloud  │◄─────────────│    Sentinel    │
     └─────────────┘   poll       │ (laptop / cron)│
     ┌─────────────┐              │                │
     │ RunPod API  │◄─────────────│                │
     └─────────────┘   poll       └────────────────┘
```

**Startup gate** (`rlmwatch preflight`) turns slow expensive failures into fast
cheap ones. Fifteen probes; the two that earn their place most are the
checkpoint round-trip (writes to the real checkpoint dir and reads it back —
prevents a sixteen-hour run that cannot save) and the RunPod write-scope check
(a read-only key serves every read happily and fails at exactly the moment the
failsafe tries to terminate).

**Watchdog** sees stack traces, `nvidia-smi`, the filesystem — and dies with the
pod, so it can never report pod death. Its probe set omits the pod-level checks
rather than offering false assurance.

**Sentinel** is coarse and survives everything. It holds the three things the
watchdog structurally cannot: the startup deadline (the only way to catch a
crash *before* `wandb.init()`), the dead-man's switch (the backstop for SIGKILL,
OOM-kill and host failure, none of which run a `finally` block), and the budget
and wall-clock caps.

**Escalation ladder**: `L0 log → L1 notify → L2 snapshot → L3 checkpoint+halt →
L4 terminate`. Nothing past L1 happens on a single unconfirmed reading, and
sustained `unknown` is promoted to `warn` and stops there — a blind monitor is
an incident, never a kill reason.

## Safety rails that are not optional

1. Set a RunPod spending limit **before** deploying. `provision.py` refuses
   without `--spend-cap-confirmed`.
2. Never launch training in a bare SSH shell — always `tmux`. `checks.py --at 2`
   verifies the session exists.
3. The pod's volume disk is deleted on terminate and there is no network volume
   by design. `finish.py` uploads to HF, verifies by reading the repo back, and
   refuses to terminate if that verification failed.
4. `stop` pauses compute and keeps billing storage; `terminate` deletes.
   Different operations, and the config and CLI both make you choose explicitly.
5. Secrets live in `.env` (gitignored) and reach the pod as environment
   variables only. Configs are committable because they interpolate `${VAR}`.

## Test suite

```
298 passed        ruff: all checks passed
```

Everything runs against in-memory fakes of both APIs — no GPU, no network, no
cloud resources, and a simulated 24-hour soak completes in about a second.
`tests/chaos/` covers the nine acceptance faults: `kill -9`, `SIGSTOP`, a W&B
network cut, a full disk, constant reward, a 10× rollout, normal completion, a
dead sentinel, and a read-only API key. Three real defects were found by those
tests and fixed rather than tested around — see the PR history.
