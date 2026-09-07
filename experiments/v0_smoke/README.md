# V0 — smoke test

**Budget: ~3 hours, under $5.** The purpose is plumbing, plus exactly one
number: **seconds per step**.

V0 cannot answer the research question and is not meant to. At 32 rollouts/step
with roughly binary reward at p≈0.24, the standard error per step is
0.43/√32 ≈ **±7.6 points** — a real 5-point gain is invisible, and reward is the
last thing to move anyway. Movement on a task like this appears somewhere around
10k–30k cumulative rollouts. V0 produces 640. **Do not read anything into V0's
reward curve.**

What V0 buys is the s/step figure that tells you whether V1 is affordable before
you book 16 hours of A100.

---

## Before anything else

| # | Do this | Verify |
|---|---|---|
| 1 | RunPod console → Billing → **set a spending limit** | limit visible in console |
| 2 | Enable low-balance notifications | — |
| 3 | `export RUNPOD_API_KEY=...` (**write scope**, not read-only) | `runpodctl gpu list` returns live prices |
| 4 | `export HF_TOKEN=...` | `huggingface-cli whoami` |
| 5 | `export WANDB_API_KEY=...` | — |

The account default cap is $80/hr, which is not a safety net. A forgotten
2× A40 pod is **~$21/day**.

`provision.py` refuses to run without `--spend-cap-confirmed`. That flag is you
asserting step 1 is done; nothing can check it for you through the API.

---

## Execution order

Each step is completed and verified before the next. **Bold rows are stop
points** — they are not checkpoints to note and move past.

| # | Task | Verify |
|---|---|---|
| 1 | Spend cap, tooling, auth | `runpodctl gpu list` returns prices |
| 2 | `provision.py --spend-cap-confirmed` | `nvidia-smi` shows 2 GPUs @ 48GB |
| 3 | `setup.sh` on the pod | `huggingface-cli download` completes |
| 4 | `smoke.toml` in place | all three "will bite" items handled |
| 5 | `strip_sub_lm_calls.py --apply` | `--verify` reports no live references |
| 6 | `rlmwatch preflight` then launch under tmux | panes for trainer, orchestrator, inference |
| 7 | **GATE: `checks.py` at T+2 / 5 / 15 / 30 min** | all four pass |
| 8 | 20 steps; record seconds/step | number written down |
| 9 | `finish.py` — upload to HF, **then** terminate | artifacts visible on HF |
| 10 | **GATE: report results, wait for human go/no-go on V1** | — |

---

## Provision

**No network volume.** It is excluded deliberately: it pins you to one
datacenter, limits GPU availability, and bills $0.07/GB/month indefinitely. Use
the pod's volume disk and push the adapter to HF before terminating.

| Field | Value |
|---|---|
| GPU | **A40 48GB × 2** ($0.44/hr each; A6000 at $0.49 if out of stock) |
| Tier | Secure Cloud |
| Template | official RunPod PyTorch, CUDA 12.x |
| Container disk | 30 GB |
| Volume disk | 100 GB → `/workspace` |
| Network volume | none |

Both GPUs in **one pod** — prime-rl needs trainer and inference on the same node.

```bash
python experiments/v0_smoke/provision.py --spend-cap-confirmed
```

It prints the pod id, the SSH command, and the hourly rate. Put the pod id in
`configs/rlm-ft-v0-smoke.yaml` (or export `RUNPOD_POD_ID`) so the monitor can
actually terminate it.

---

## The three things that will bite

These are in `smoke.toml` and `strip_sub_lm_calls.py` already. They are listed
here because each one has a characteristic failure that looks like something
else:

1. **`flash_attention_3` will not build on Ampere.** The example config assumes
   A100/H100. A40 is Ampere. This is the most likely first crash, and it happens
   during model init, not at step 1.

2. **Qwen3-8B is hybrid-thinking.** Left enabled, trajectories burn their token
   budget on reasoning traces before ever touching the REPL. The run looks alive
   and produces nothing. Disabled via `extra_body: {"enable_thinking": false}`.

3. **Sub-LM calls must go from the REPL globals *and* the system prompt.**
   Removing them from only one is worse than removing them from neither: a
   leftover prompt mention means the model keeps calling `llm_query`, every
   rollout dies on `NameError`, and from outside the run looks perfectly
   healthy while scoring zeros. `strip_sub_lm_calls.py --verify` checks both.

---

## The four startup checks

```bash
python experiments/v0_smoke/checks.py --at 2     # inference alive
python experiments/v0_smoke/checks.py --at 5     # read one trajectory BY EYE
python experiments/v0_smoke/checks.py --at 15    # first optimizer step
python experiments/v0_smoke/checks.py --at 30    # weight sync
```

Run them in order. **Stop at the first failure.**

**T+5 is the one people skip, and the one that matters.** It prints a trajectory
for you to read. `--at 5` cannot pass or fail on its own — it shows you the code
and stdout and asks five questions. Answer them by looking.

**T+30 is the classic silent failure.** The trainer updates, the inference server
never receives new weights, and you sample from the base model forever while the
loss curve looks perfect. `checks.py --at 30` looks for the broadcast log line;
if it is absent it samples the same prompt at step 0 and step 15 and tells you
whether the outputs diverged.

---

## Exit criteria

1. Valid REPL trajectories with sub-calls removed
2. Reward function returned varied scores
3. Weights synced to the inference server
4. LoRA adapter on disk
5. **Measured seconds per step** ← the actual deliverable

Multiply (5) out before booking V1. 250 steps at the measured rate, on 2× A100
at $1.39 each, against a ~$60 total budget.

---

## Getting the artifact off the pod

**The volume disk dies with the pod.** `finish.py` uploads first and only then
offers to terminate, and it refuses to terminate if the upload did not succeed.

```bash
python experiments/v0_smoke/finish.py --hf-repo <you>/rlm-smoke-8b
```

`stop` preserves the disk but keeps billing it. `terminate` deletes everything
not on a network volume, and there is no network volume here by design.

---

## When to stop and escalate

- The same command fails twice
- A gate check fails and the cause is not obvious from the logs
- Reward standard deviation is zero
- No weight-sync log line appears
- Projected cost from the measured step time exceeds the budget
- You are tempted to change a config value not listed as changeable

A third attempt at the same failing command is never the right move.

```
BLOCKED at task <n>
Command:  <exact command>
Error:    <last 20 lines>
Tried:    <what you attempted, twice>
Guess:    <your hypothesis, or "none">
Cost so far: $<x>   Pod still running: yes/no
```

Say plainly if you do not know. A wrong guess stated confidently costs more than
"none".

---

## Intervention log

Part of the point of this run is measuring how far an agent gets unaided. Every
time a human has to act:

```bash
python experiments/v0_smoke/interventions.py \
  --task 3 --category DEPS \
  --tried "uv pip install -e . failed on flash-attn build" \
  --human-did "installed prebuilt wheel"
```

Do not omit entries to look better. **The log is the experiment, not a
scorecard.**
