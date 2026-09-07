# rlm_fine_tune

RL fine-tuning of a reasoning LM on RunPod GPUs, with the monitoring layer that
keeps an unattended run from quietly burning the budget.

Two deliverables live here, built in this order:

| Directory | What it is |
|---|---|
| `rlmwatch/` | Reusable monitoring library — startup gate, in-pod watchdog, external sentinel, escalation ladder. Built and fault-tested *before* it is wired to anything. |
| `experiments/v0_smoke/` | The V0 smoke test: provision 2x A40, run 20 steps of RLM context-management RL, measure seconds/step, upload the adapter, terminate. |

Specs: [`_instructions.md`](_instructions.md) (the experiment) and
[`_monitoring_instructions.md`](_monitoring_instructions.md) (the monitor).
Both are normative — code follows them, not the other way round.

## The one-paragraph version

**Experiment.** With sub-LM calls stripped out, does RL improve a model's use of
a Python REPL over long context? V0 (this repo's current scope) only proves the
plumbing and produces one number: seconds per step. V1 — the run that can
actually answer the question — is not started until a human signs off on V0.

**Monitor.** RunPod tells you nothing when your script dies, hangs, or finishes
(the container restarts and the meter keeps running). W&B's heartbeat comes from
its own service process, so a deadlocked training loop shows `Running` forever.
`rlmwatch` closes that gap with two independent observers: a watchdog inside the
pod that can see stack traces but dies with the pod, and a sentinel outside it
that survives everything and holds the dead-man's switch.

## Quick start

```bash
cp .env.example .env          # fill in RUNPOD_API_KEY, WANDB_API_KEY, HF_TOKEN
uv venv && uv pip install -e ".[dev]"
pytest                        # library + chaos suite, no cloud resources touched
```

Nothing in this repo provisions a GPU until you run the scripts in
`experiments/v0_smoke/` explicitly, and those refuse to start without a
spend cap on the account.

## Safety rails that are not optional

1. Set a RunPod spending limit **before** deploying. The account default is
   $80/hr, which is not a safety net. A forgotten 2x A40 pod is ~$21/day.
2. Never launch training in a bare SSH shell — always `tmux`.
3. The pod's volume disk is deleted on terminate. Artifacts go to HF *first*.
4. `stop` pauses compute and keeps billing storage; `terminate` deletes.
   They are different operations and the config makes you choose explicitly.
5. Secrets live in `.env` (gitignored) and reach the pod as environment
   variables only.
