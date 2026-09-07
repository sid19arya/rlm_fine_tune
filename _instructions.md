# RLM context-management RL on RunPod — V0 and V1

---

# PART 0 — AGENT BRIEF

**Read this whole section before running any command.**

You are configuring and running a reinforcement-learning experiment on RunPod GPUs. Work through **V0 first**. Do not start V1 until a human confirms V0 passed.

## Your scope

| You may | You may not |
|---|---|
| Create, start, stop and terminate pods | Spend beyond the stated budget without asking |
| Install packages, edit config files, run training | Deploy anything larger than the specified GPU count |
| Read logs, diagnose failures, retry a failed step twice | Change `max_steps`, batch size, or model to "make it work" |
| Report progress and escalate | Delete a pod before artifacts are uploaded |

**If a step fails twice with the same error, stop and escalate.** Do not loop. Do not improvise a workaround that changes the experiment. A third attempt at the same failing command is never the right move.

## Hard rules

1. **Set the spend cap first.** Before deploying anything: RunPod console → Billing → spending limit. Also enable low-balance notifications. A forgotten 2-GPU pod costs ~$21/day.
2. **Never launch training in a bare SSH shell.** Always `tmux`. A dropped connection kills an unwrapped run.
3. **The pod's volume disk is deleted on terminate.** Upload adapters and logs *before* terminating. `stop` preserves the disk but keeps billing.
4. **Do not create a network volume.** It is deliberately excluded — it pins the datacenter and bills monthly.
5. **Never commit or echo the API key or HF token.** Use environment variables only.
6. **Stop at every gate.** Gates are marked. They are not optional checkpoints to note and move past.

## Prerequisites (human does these once)

- RunPod account with credit and a spending limit set
- `RUNPOD_API_KEY` from console → Settings → API Keys
- Hugging Face account and `HF_TOKEN` (needed to upload the adapter at the end)
- SSH keypair registered to the RunPod account

## Tooling setup (agent does this, locally, before touching GPUs)

```bash
npx skills add runpod/runpod-plugins-official     # router + runpod skills
curl -sSL https://cli.runpod.net | bash           # runpodctl
export RUNPOD_API_KEY=<key>
export HF_TOKEN=<token>
runpodctl doctor                                  # verify auth
```

Confirm the wiring works before proceeding: run `runpodctl gpu list` and check that A40 and A100 appear with live prices. If that fails, stop — nothing downstream will work.

## Execution order

Complete each task fully, verify, then move to the next. **Bold gates are stop points.**

| # | Task | Verify | Section |
|---|---|---|---|
| 1 | Set spend cap, install tooling, auth | `runpodctl gpu list` returns prices | above |
| 2 | Deploy V0 pod: 2× A40, 30GB container, 100GB volume, no network volume | `nvidia-smi` shows 2 GPUs @ 48GB | V0 Provision |
| 3 | Install uv, rlm, prime-rl; pre-download Qwen3-8B | `huggingface-cli download` completes | V0 Setup |
| 4 | Edit config → `smoke.toml` | all three "will bite" items handled | V0 Config |
| 5 | Remove sub-LM calls from REPL globals **and** system prompt | `grep -rn "llm_query" training/src/` shows no live references | V0 Config |
| 6 | Launch under tmux | panes for trainer, orchestrator, inference | V0 Launch |
| 7 | **GATE: four startup checks at T+2/5/15/30min** | all four pass | V0 Launch |
| 8 | Run 20 steps, record seconds/step | number written down | V0 Exit |
| 9 | Upload adapter + logs to HF, terminate pod | artifacts visible on HF | V0 Exit |
| 10 | **GATE: report V0 results, wait for human go/no-go** | — | — |
| 11 | Deploy V1 pod: 2× A100 80GB | `nvidia-smi` shows 2 GPUs @ 80GB | V1 |
| 12 | Repeat 3–6 with V1 config | — | V1 |
| 13 | At step ~25, kill and resume from checkpoint on purpose | run continues from checkpoint | V1 |
| 14 | **GATE at step 100: apply the decision table** | continue or stop | V1 Gate |
| 15 | Complete 250 steps, upload adapter | — | V1 |
| 16 | Run three-cell eval, paired, n≥150 | — | V1 Eval |

## What "stuck" looks like — escalate on any of these

- The same command fails twice
- A gate check fails and the cause isn't obvious from the logs
- Reward standard deviation is zero
- No weight-sync log line appears
- Projected cost from measured step time exceeds the stated budget
- You are tempted to change a config value not listed as changeable

## Escalation format

```
BLOCKED at task <n>
Command:  <exact command>
Error:    <last 20 lines>
Tried:    <what you attempted, twice>
Guess:    <your hypothesis, or "none">
Cost so far: $<x>   Pod still running: yes/no
```

Say plainly if you don't know. A wrong guess stated confidently costs more than "none".

## Intervention log — keep this from the first command

Part of the point of this run is measuring how far an agent gets unaided. Append to `interventions.jsonl` every time a human has to act:

```json
{"ts":"<iso8601>","task":<n>,"category":"AUTH|PROVISION|ORDERING|DEPS|CONFIG|LONGRUN|DIAGNOSE","tried":"...","human_did":"...","recoverable":true}
```

Do not omit entries to look better. The log is the experiment, not a scorecard.

---

# PART 1 — THE EXPERIMENT

Testing one question: **with sub-LM calls removed, does RL improve a model's use of a Python REPL over long context?** If yes, recursion isn't the driver.

Two levels. V0 proves the plumbing. V1 is the smallest run that can actually answer the question.

| | **V0 — smoke** | **V1 — movement** |
|---|---|---|
| Question | does it run? | does held-out eval move? |
| Model | Qwen3-8B | Qwen3-8B |
| Hardware | 2× A40 48GB | 2× A100 80GB (or 4× to halve wall clock) |
| Rollouts/step | 32 | 128 |
| Steps | 20 | 250 (gate at 100) |
| Wall clock | ~3h incl. setup | ~16h + 4h eval |
| Cost | **~$3** | **~$60** |
| Success | valid trajectories, weights sync, s/step measured | held-out delta outside the noise band |

Do not run V1 until V0 passes. Do not read anything into V0's reward curve.

---

## Why V0 can't answer the question

At 32 rollouts/step with roughly binary reward at p≈0.24 (base Qwen3-8B scores ~24 on OOLONG in the RLM harness), standard error per step is 0.43/√32 ≈ **±7.6 points**. A real 5-point gain is invisible. And reward is the *last* thing to move anyway.

Count **cumulative rollouts**, not steps. Movement on a task like this appears around 10k–30k rollouts. V0 gives you 640.

---

# V0 — Smoke test

**Budget: ~3 hours, under $5.** Purpose is plumbing, plus one number: seconds per step.

## Provision

**Skip the network volume.** It pins you to a datacenter, limits GPU availability, and bills $0.07/GB/month indefinitely. Use the pod's volume disk and push the adapter to HF before terminating.

| Field | Value |
|---|---|
| GPU | **A40 48GB × 2** ($0.44/hr each; A6000 at $0.49 if out of stock) |
| Tier | Secure Cloud |
| Template | official RunPod PyTorch, CUDA 12.x |
| Container disk | 30 GB |
| Volume disk | 100 GB → `/workspace` |
| Network volume | none |

Both GPUs in **one pod** — prime-rl needs trainer and inference on the same node. Set a spend cap and low-balance alerts before deploying; a forgotten 2×A40 pod is ~$21/day.

## Setup (~25 min, mostly download)

```bash
ssh root@<ip> -p <port>
nvidia-smi                            # exactly 2 GPUs, 48GB each, idle
df -h /workspace                      # ~100G

export HF_HOME=/workspace/hf          # keep weights OFF the 30GB container disk
echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc

cd /workspace
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
git clone https://github.com/alexzhang13/rlm && cd rlm
uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e .
cd /workspace && git clone https://github.com/PrimeIntellect-ai/prime-rl   # install per its README

huggingface-cli download Qwen/Qwen3-8B     # pre-download; don't do this during step 1
```

## Config

Copy `training/configs/rlm-qwen3-30b-example.toml` → `smoke.toml`:

```toml
inference_gpu_ids = [0]
trainer_gpu_ids   = [1]
max_steps = 20

[model]
name = "Qwen/Qwen3-8B"

[trainer.model]
attn = "flash_attention_2"     # NOT fa3 — Hopper only, A40 is Ampere
impl = "auto"

[trainer.model.experimental.lora]
rank = 32
alpha = 64
dropout = 0.0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

[orchestrator]
batch_size = 8
rollouts_per_example = 4       # 32 rollouts/step

[orchestrator.train.sampling]
max_completion_tokens = 2048
```

**Three things that will bite:**

1. `flash_attention_3` won't build on Ampere. The example config assumes A100/H100. Most likely first crash.
2. Qwen3-8B is hybrid-thinking. Disable it (`extra_body: {"enable_thinking": false}`) or trajectories burn their budget on reasoning traces before touching the REPL.
3. **Strip sub-LM calls from both the REPL globals and the system prompt** in `training/src/rlm_train/`. A leftover prompt mention means every rollout dies on `NameError` while looking healthy from outside.

VRAM: inference GPU = 16GB weights + ~28GB KV ≈ 11 concurrent 16k sequences. Trainer = 16GB frozen + 0.4GB adapter + activations. Both fine.

## Launch and verify

```bash
tmux new -s rlm                       # never a bare SSH shell
uv run rl @ training/configs/smoke.toml
```

Four checks, in order. Stop at the first failure.

**T+2min — inference alive.** `curl -s localhost:8000/v1/models | jq` returns your model id.

**T+5min — read one trajectory by eye.** The check people skip, and the one that matters:
```bash
jq -r '.iterations[] | .code, .stdout' /workspace/logs/rlm_*.jsonl | head -40
```
Confirm the ` ```repl ` block parsed, code ran without traceback, `context` was in scope at expected length, **no mention of `llm_query`**, and the loop ended via `answer["ready"]` rather than exhausting iterations.

**T+15min — first optimizer step.** Loss present, grad norm nonzero, **reward std > 0**. Zero variance means no gradient signal exists at all. Kill it.

**T+30min — weight sync.** The classic silent failure: trainer updates, inference server never receives new weights, you sample from the base model forever while the loss curve looks perfect. Find the broadcast log line. If absent, sample the same prompt at step 0 and step 15 and confirm outputs diverged.

## V0 exit criteria

1. Valid REPL trajectories with sub-calls removed
2. Reward function returned varied scores
3. Weights synced to the inference server
4. LoRA adapter on disk
5. **Measured seconds per step**

(5) is the actual deliverable. Multiply it out before booking V1.

Get your artifact off the pod before terminating — volume disk dies with it:
```bash
huggingface-cli upload <you>/rlm-smoke-8b /workspace/outputs/<adapter>
```

---

# V1 — Movement run

**Budget: ~20 hours, ~$60.** Purpose is a held-out delta you can defend.

## What changes from V0

| | V0 | V1 | Why |
|---|---|---|---|
| GPU | 2× A40 | **2× A100 80GB** ($1.39 ea) | ~2× throughput + KV headroom for the same money per unit work |
| `batch_size` | 8 | **16** | |
| `rollouts_per_example` | 4 | **8** | 128/step → noise band ±3.8 vs ±7.6 |
| `max_steps` | 20 | **250** | ~32k cumulative rollouts |
| `max_completion_tokens` | 2048 | **4096** | match reference conditions |
| Checkpoints | — | **every 25 steps** | resume matters over 16h |

Raising `rollouts_per_example` is nearly free variance reduction — same compute per rollout, much cleaner per-step signal.

**Optional:** 4× A100 (2 inference / 2 trainer) roughly halves wall clock at similar total cost. Worth it if you're waiting on the result.

Before going unattended: kill the run at step ~25 and **restart from checkpoint on purpose.** Pods get migrated and come back with zero GPUs. Discover broken resume at hour 1, not hour 14.

## What moves when

Behavioral metrics have far better signal-to-noise than reward — they're measured per-turn, not per-episode:

| Signal | Visible by |
|---|---|
| REPL error rate falling | **step 10–25** |
| Fewer `max_iterations` timeouts | step 15–30 |
| Output length stabilizing | step 10–20 |
| Mean REPL turns per rollout | step 20–40 |
| **Reward mean (10-step EMA)** | **step 50–150** |
| Held-out eval delta | step 150–400 |

If REPL error rate is falling by step 25, learning is happening even with a flat reward curve. If it's *not* falling by step 40, something is broken — bad prompt, unparseable blocks, or a rubric that doesn't discriminate.

Plot reward as a 10-step EMA with the noise band drawn on. Raw per-step reward at n=128 still looks like static.

## Gate at step 100

| Observation | Action |
|---|---|
| Reward EMA clearly rising | continue to 250 |
| Reward flat, but REPL errors and turn count moved | continue to 250 — format learned first |
| Nothing moved at all | **stop.** Debug the rubric. More steps won't help |

## Degeneracy watch

Without sub-calls, the lazy strategy is `print(context[:4000])` then answer from the truncated slice — ignoring the REPL entirely while still scoring passably. Track mean REPL turns and the fraction of rollouts running any non-trivial operation (regex, split, loop, aggregate). **If those fall while reward rises, you are not measuring context management.** Pre-register this as a kill criterion now, not after the eval.

## Eval protocol

Serve base + adapter under vLLM (`--enable-lora --max-lora-rank 64`), then three cells:

| Cell | Setup |
|---|---|
| A0 | base, raw long context, no REPL |
| A | base, REPL-only harness |
| C | adapter, REPL-only harness |

`A − A0` = harness effect. **`C − A` = your answer.**

**Held-out:** OOLONG `trec_coarse` @ 132k and OOLONG-Pairs @ 32k, plus LongBench v2 Code repo QA as the cross-family transfer test. Train on the OOLONG-Spam split only.

Two things that decide whether the result is defensible:

- **Use identical items across all three cells and report the paired delta.** Paired comparison removes item difficulty from the variance and tightens the CI dramatically. Unpaired at n=50, SE is ~6.5 points and a 10-point delta is barely 1.5σ — not a result.
- **n ≥ 150 per env.** The reference run used n=50 for trec_coarse and n=20 for Pairs; at those sizes, small deltas are noise.

Report **numeric and non-numeric splits separately.** The reference run's gains were entirely non-numeric (60.5 vs 42.1 base) with numeric flat near zero (4.9 vs 7.3). If programmatic context management is the mechanism, numeric is exactly where Python should win. If it still doesn't move here, that's the most informative result available — it suggests models aren't using the interpreter computationally at all.

Also run a **short-context control** (GSM8K or an MMLU subset) on base vs adapter *outside* the harness. Narrow RL can quietly degrade general capability, and you want to find that yourself.

## Reading the result

- **`C − A` positive, transfers to LongBench CodeQA** → RL teaches harness use. Recursion isn't the driver. Next axis is tools, not depth.
- **`C ≈ A`** → harness does the work, weights add nothing here. Re-run *with* sub-calls to test whether delegation was the real mechanism.
- **`C − A` positive but only on the training family** → learned the env, not the skill. Still an answer; report it plainly.

## Cost

| | |
|---|---|
| V0: 2× A40 × 3h | ~$2.60 |
| V1: 2× A100 × 16h | ~$45 |
| V1 eval: 2× A100 × 4h | ~$11 |
| **Total** | **~$60** |

For calibration: the reference 30B run was ~a day on 8×A100 — roughly 250–350 steps at 4–6 min/step. That's what a *publishable* delta cost. You're buying the same step count at a twentieth of the compute by dropping to 8B and cutting the sub-call fan-out.