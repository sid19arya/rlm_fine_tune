# monitor.md — Monitoring spec for RLM fine-tuning on RunPod

**Audience:** an implementing agent (Claude Code or equivalent).
**Deliverable:** a reusable Python library (`rlmwatch`), then a thin wiring layer that attaches it to a specific RLM fine-tuning run.

**Order of work is mandatory.** Build and test the library against injected faults *first*. Only then wire it to the live experiment. Do not write bespoke monitoring glue inside the training script — anything experiment-specific goes in config, not in code.

---

## 0. Why this exists (platform facts to design against)

These are properties of RunPod and W&B, verified. The library exists because of them.

**RunPod tells you almost nothing about your workload.**

| Situation | What RunPod does |
|---|---|
| Planned host maintenance | Emails you in advance; you aren't billed for downtime |
| Unplanned host outage | Notifies you *after* it starts, once identified |
| Your script raises and dies | Nothing |
| Your script hangs | Nothing |
| Your script finishes | Nothing — and the container **restarts**, meter still running |
| GPU idle at 0% for 6 hours | Nothing |

Billing is per-second from the moment the pod is *provisioned*, not from when your workload starts. There is no built-in alerting for custom metrics. Storage keeps accruing on stopped pods. Default account spend cap is $80/hr, which is not a safety net.

**W&B covers part of the gap but has one specific blind spot.**

- Auto-logs GPU utilization, memory, temperature, power as system metrics — no instrumentation needed.
- Run states: `Running`, `Finished`, `Crashed`, `Failed`, `Killed`, `Pending`.
- `Crashed` fires when heartbeats stop from the process that called `wandb.init()`.
- **The heartbeat comes from wandb's internal service process, not your training loop.** A deadlocked loop keeps heartbeating. A hung run shows `Running` forever.
- Converse failure: a long network outage can mark a live run `Crashed`.

**Consequences that shape the design:**

1. Run state alone is neither necessary nor sufficient. Last-progress-age is the real liveness signal.
2. Any observer running *inside* the pod dies with the pod. Any observer running *outside* is too slow and coarse to catch everything. **You need both.**
3. Process exit does not stop billing. Termination must be explicit.

---

## 1. Architecture: two independent observers

```
   ┌─────────────────────── RunPod pod ────────────────────────┐
   │                                                            │
   │   train.py                                                 │
   │     ├─ rlmwatch.startup.gate()      ← runs before training │
   │     ├─ rlmwatch.Watchdog(...)       ← in-process thread    │
   │     └─ training loop → wandb.log(...)                      │
   │                                                            │
   └────────────────────────────────────────────────────────────┘
              │ wandb.log                    ▲ stop / terminate
              ▼                              │
       ┌─────────────┐              ┌────────────────┐
       │  W&B cloud  │◄─────────────│    Sentinel    │
       └─────────────┘   poll       │ (laptop / cron │
                                    │  / cheap VM)   │
       ┌─────────────┐              │                │
       │ RunPod API  │◄─────────────│                │
       └─────────────┘   poll       └────────────────┘
```

**Watchdog (in-pod).** Sub-second visibility. Sees stack traces, `nvidia-smi`, the filesystem. Can checkpoint before dying. **Dies with the pod, so it can never report pod death.**

**Sentinel (external).** Survives everything. Sees pod status and W&B state. Coarse (minutes), and cannot see inside the process. **This is the only component that can catch a hung or dead pod, so it must not run on the pod.**

Every failure mode in §5 must be assigned to one or both. A mode assigned to neither is an unhandled mode.

---

## 2. Library layout

```
rlmwatch/
  __init__.py
  config.py        # RunConfig dataclass + YAML load/validate
  clients/
    runpod.py      # pod status, log stream, stop, terminate
    wandb.py       # run state, metric history, system metrics, last-progress-age
  probes/
    base.py        # Probe protocol, Verdict
    startup.py     # preflight probes (§3)
    liveness.py    # progress / util / state probes (§4.1)
    health.py      # metric-shape probes (§4.2)
    cost.py        # budget probes (§4.3)
    rlm.py         # RLM-specific probes (§5)
  watchdog.py      # in-process supervisor thread
  sentinel.py      # external poller (long-running or cron one-shot)
  actions.py       # escalation ladder (§6)
  diagnostics.py   # snapshot bundle: py-spy, nvidia-smi, logs, metrics
  notify.py        # Slack / webhook / email sinks
  cli.py           # rlmwatch preflight | watch | snapshot | kill
tests/
  fakes/           # fake RunPod + W&B servers
  chaos/           # fault injection harness (§8)
```

### Core interfaces

Keep these small and stable; everything else is replaceable.

```python
Status = Literal["ok", "warn", "fail", "unknown"]

@dataclass(frozen=True)
class Verdict:
    probe: str
    status: Status
    detail: str                    # human-readable, one line
    evidence: dict                 # raw numbers that produced the verdict
    at: datetime

class Probe(Protocol):
    name: str
    def check(self, ctx: Context) -> Verdict: ...

@dataclass
class Context:
    cfg: RunConfig
    wandb: WandbClient
    runpod: RunPodClient
    phase: str | None              # current training phase, if reported
    started_at: datetime
```

Rules:
- Probes are **pure predicates**. They never notify, never terminate, never mutate. Actions are decided by the escalation ladder in §6.
- A probe that cannot determine an answer returns `unknown`, never `ok`. `unknown` sustained past `unknown_tolerance` escalates to `warn` — a monitor that has gone blind is itself an incident.
- Every probe carries `evidence` so alerts are actionable without opening a browser.

### Client requirements

**`clients/runpod.py`**
- `status(pod_id)` — pod state, GPU type/count, uptime.
- `logs(pod_id, since=None, tail=None, source=None)` — wraps `GET /v2/pods/{id}/logs`, Server-Sent Events, payload `{"source","line","ts"}`. Event ids are timestamps; support resume via `Last-Event-ID`. Expose as an iterator **and** a bounded `tail_lines(n)` — the agent must never stream logs continuously into an LLM context.
- `stop(pod_id)` — `POST /v1/pods/{id}/stop`. Compute paused, storage still billed.
- `terminate(pod_id)` — `DELETE /v1/pods/{id}`. Deletes everything not on a network volume.
- Retry with backoff on 5xx. Treat 403 as a config error (read-only key) and fail loudly at startup, not at kill time.

**`clients/wandb.py`**
- `state(run_path)` → run state.
- `last_progress_age(run_path)` → seconds since the newest logged step (`run.summary["_timestamp"]`). **The single most important signal in this library.**
- `metric_window(run_path, key, n)` → recent history for trend probes.
- `system_metrics(run_path)` → `system/gpu.*.gpu`, `gpu.*.memoryAllocated`, temperature, power.
- Never trust a single poll. Confirm any `fail` with a re-poll after `confirm_delay` before escalating; network blips must not terminate a healthy $30/hr pod.

---

## 3. Phase 1 — Startup gate

Runs inside the pod, **before** the first training step. Purpose: convert slow expensive failures into fast cheap ones. A misconfigured run should die in 90 seconds, not after six hours when the first checkpoint write fails.

`rlmwatch.startup.gate(cfg)` runs probes in order, short-circuits on first `fail`, and on failure **terminates the pod** (configurable to stop) after emitting diagnostics. It must never leave a failed pod running.

**Hardware**
- GPU count matches `cfg.expect.gpu_count`; type matches `cfg.expect.gpu_type`.
- `torch.cuda.is_available()`; `device_count()` agrees with `nvidia-smi`.
- Free VRAM per device ≥ `cfg.expect.min_free_vram_gb`.
- `nvidia-smi -q` shows no uncorrectable ECC errors, no pending retired pages; scan dmesg/system logs for Xid events. Fail on Xid, warn on correctable ECC.
- Multi-GPU: NCCL all-reduce smoke test on a small tensor, wrapped in a hard timeout. A NCCL hang here is common and must not become an indefinite block.

**Storage**
- Network volume mounted at the expected path and writable.
- Free space ≥ `cfg.expect.min_disk_gb`, sized against `checkpoint_size × keep_n`.
- **Checkpoint round-trip test:** write a dummy tensor to the real checkpoint directory, read it back, delete it. This single probe prevents the worst outcome in the whole system — a long run that cannot save.

**Model and data**
- Base model and tokenizer load; parameter count matches expectation.
- Dataset resolves; first batch materializes; shapes and dtypes as expected.
- Sequence lengths in the first N batches fit `max_seq_len` (see §5, length growth).
- Resume path, if set, loads and reports the step it resumed from.

**Observability self-test** — the monitor must prove itself before the run trusts it:
- `wandb.init()` succeeded and the run is visible via the public API from outside the process.
- A canary metric logged in-process is readable back through `WandbClient` within `confirm_delay`.
- Notification sink reachable (post a startup message to Slack; its absence is itself the signal).
- RunPod API key has **write** scope — verify by a benign authenticated write, not by assuming. Discovering a read-only key at kill time defeats the entire failsafe layer.

**Warm-up**
- One full training step completes and logs.
- RL only: one full rollout → score → update cycle completes; record its duration as the baseline for phase-aware staleness thresholds (§4.1).
- Generation backend (vLLM/SGLang) health endpoint responds, if applicable.

On success, log `startup_ok = 1` plus a `startup_duration_s` and the measured phase baselines to W&B. The sentinel treats absence of `startup_ok` within `cfg.startup.deadline_s` of pod creation as a failure and terminates the pod — this covers crashes that occur before `wandb.init()`, which W&B can never see.

---

## 4. Phase 2 — Steady-state monitoring

### 4.1 Liveness — is anything happening?

| Signal | Source | Owner | Fail condition |
|---|---|---|---|
| Progress age | W&B `_timestamp` | Sentinel + Watchdog | `> stall_threshold(phase)` |
| Run state | W&B | Sentinel | `Crashed` / `Failed` / `Killed` |
| GPU utilization | W&B system metrics | Sentinel | mean `< 5%` over 10 min while state is `Running` |
| Pod status | RunPod API | Sentinel | not `RUNNING` while W&B says `Running` (or vice versa) |
| Step counter | in-process | Watchdog | no increment in `stall_threshold(phase)` |

**Phase-aware staleness is not optional for RL.** A GRPO rollout can run many minutes with nothing logged. A flat threshold either false-alarms constantly or is set so high it is useless.

Requirement: the training script emits a `phase` string (`rollout` / `score` / `update` / `eval` / `checkpoint`) and a monotonic `heartbeat` counter **inside** long phases. Thresholds are then per-phase, derived from the warm-up baseline measured in §3:

```yaml
stall_threshold:
  default:  300
  rollout:  auto   # 4× measured baseline, floor 600s
  eval:     auto
  checkpoint: 900
```

Disagreement between observers is itself a signal. Pod `RUNNING` + W&B `Crashed` means the process died but the container survived — the expensive silent case, and the single most valuable alert this library produces.

### 4.2 Progress — is it going anywhere?

- Throughput (tokens/s or steps/hr) vs. the warm-up baseline; warn on sustained degradation past `throughput_degradation_pct` (thermal throttling, dataloader starvation, memory fragmentation, KV-cache pressure).
- Projected completion vs. `cfg.budget.max_wall_clock_h`.
- Loss finite. `NaN`/`Inf` is an immediate `fail`, not a warn.
- Gradient norm within `[grad_norm_min, grad_norm_max]`; sustained excursion warns.

### 4.3 Cost

- Accrued spend = elapsed × `cfg.cost.hourly_rate`, plus storage.
- Warn at `budget_warn_pct` (default 75%), hard stop at `budget_max_usd`.
- **Cost-per-progress**: dollars per 1% of planned steps. Rising sharply means you are paying more for less — catches slow degradation that no absolute threshold sees.
- Report projected total spend in every alert. An operator deciding whether to intervene at 2am needs the number in the message.

---

## 5. Phase 3 — RLM fine-tuning specialization

> **Assumption to confirm before implementing.** "RLM fine-tuning" is read here as fine-tuning a *reasoning* language model, covering both regimes below. Set `regime: sft | rl` in config; the library loads the matching probe set. If RLM means something else in your setup, only this section changes — §§1–4 and 6–8 are regime-independent.

### 5.1 Regime `sft` — supervised fine-tuning on reasoning traces

| Metric | Watch for | Action |
|---|---|---|
| `train/loss` | NaN/Inf; plateau with zero movement over `plateau_window` | fail / warn |
| `train/grad_norm` | spikes ≫ baseline (bad batch, LR too high); collapse to ~0 (dead training) | warn |
| Token accuracy on reasoning spans | flat while total loss falls — model learning formatting, not reasoning | warn |
| Sequence length p95 | approaching `max_seq_len` → truncation silently discarding conclusions | warn |
| Packing efficiency | dropping → wasted compute per step | warn |
| Held-out eval | non-monotone decline over `eval_patience` evals → overfitting | warn, suggest halt |

### 5.2 Regime `rl` — GRPO / PPO / policy-gradient post-training

These are the modes that actually eat budgets, because the job looks perfectly alive throughout.

| Metric | Failure mode | Detection | Action |
|---|---|---|---|
| `reward/std` within group | **Zero-variance collapse.** All completions in a GRPO group score identically → advantage is zero → gradient is zero. Training is a no-op but every dashboard looks normal. | `reward_std < eps` for `N` consecutive steps | **fail** — this is the highest-value probe in the file |
| `reward/mean` | Reward hacking: reward climbs while eval quality falls | reward up + KL up + eval flat/down | warn |
| `kl/ref` | KL blowup — policy has left the reference distribution; outputs degenerate | `> kl_max`, or slope over window | fail |
| `policy/entropy` | Entropy collapse — deterministic policy, exploration dead, learning stops | below `entropy_min` or monotone decline over window | warn → fail |
| `completion/length` | Length hacking (padding to farm reward) or collapse to trivial answers | mean outside `[len_min, len_max]`, or trend | warn |
| Format/parse success rate | Model stops emitting valid reasoning delimiters or tool-call syntax; scorer silently returns zeros | `< format_success_min` | fail |
| `clip_fraction` (PPO) | Sustained high → steps too large, updates being discarded | `> clip_max` | warn |
| Rollout duration | Generation backend degrading or wedged; the most common RL hang | `> 4×` warm-up baseline | fail |
| Generation backend health | vLLM/SGLang process dead or unresponsive while trainer waits forever | health endpoint + rollout-phase heartbeat | fail |
| VRAM headroom | Policy + reference + KV cache; OOM arrives *later* in training as sequences lengthen, not at step 0 | free VRAM trending toward zero | warn early |
| Reward-model latency/errors | External scorer rate-limited or down; rewards silently degrade | error rate, p99 latency | fail |

**Design note.** Distinguish *stalled* (nothing moving) from *degenerate* (moving toward a worthless model). RL's characteristic failure is the second: full GPU utilization, steady step rate, healthy-looking loss, and a policy collapsing into reward hacking. Only §5.2's shape probes catch it. Liveness monitoring alone gives false confidence here.

---

## 6. Failsafes — escalation ladder

Each level is separately configurable per probe. Never jump straight to termination on a first `fail`; always re-poll after `confirm_delay`.

**L0 — Log.** Structured record. No human notified.

**L1 — Notify.** Slack/webhook with probe name, evidence, current spend, projected spend, and a direct W&B run link.

**L2 — Snapshot** (`diagnostics.snapshot()`). Assemble a bundle and upload as a W&B artifact:
- `py-spy dump --pid <trainer> --nonblocking` for every Python process. **This is what turns a mystery hang into a stack trace** — a hung run that is killed without a py-spy dump has taught you nothing and will hang again.
- `nvidia-smi -q` full output, plus per-process memory.
- `py-spy` on the generation backend too, in the RL case.
- Last 500 log lines from the RunPod stream (bounded, never unbounded).
- Last 50 steps of all tracked metrics.
- `dmesg | tail` for Xid and OOM-killer evidence.

**L3 — Checkpoint and halt.** Signal the trainer to save at the next safe point, `wandb.finish()`, then stop. Requires a cooperative trainer exposing a `request_checkpoint()` hook. Bounded by `checkpoint_timeout_s`, after which escalate to L4 regardless.

**L4 — Terminate.** `wandb.alert()` → `wandb.finish()` → RunPod terminate. Order matters: metrics flush before the pod dies.

### Mandatory guards

**Self-termination on every exit path.** Wrap training so that *no* exit leaves a billing pod:

```python
signal.signal(signal.SIGTERM, lambda *_: shutdown("SIGTERM"))
try:
    train()
    shutdown("completed")
except Exception as e:
    shutdown(f"crashed: {e!r}")
```

`shutdown()` must, in order: request checkpoint → `wandb.alert` → `wandb.finish` → `runpodctl remove pod $RUNPOD_POD_ID` (or the REST `DELETE`).

Two things the implementer must get right:

- **A clean exit does not stop billing.** Pods restart after the entrypoint ends. Termination must be explicit.
- **`stop` and `terminate` are different.** Stop retains storage and keeps billing it; terminate deletes everything not on a network volume. Default to terminate, and require a network volume at `/workspace` so terminate is safe. Make this an explicit config choice, never an implicit default.

**Dead-man's switch (external, mandatory).** The sentinel terminates the pod if it has heard nothing for `dead_mans_timeout`. This is the backstop for SIGKILL, OOM-kill, and host failure, none of which run your `finally` block. Without it every other failsafe is best-effort.

**Budget cap and wall-clock cap.** Unconditional. Enforced by the sentinel, so they survive the pod becoming unresponsive.

**Sentinel liveness.** If the sentinel itself dies, nothing is watching. Register it with an external dead-man's-switch service (Healthchecks.io or equivalent) so its own silence pages you. Do not build a fourth layer to watch the third.

---

## 7. Configuration

Single YAML, validated at load, no monitoring logic in the training script.

```yaml
run:
  name: rlm-ft-grpo-001
  regime: rl                        # sft | rl
  wandb: entity/project/run-id
  pod_id: ${RUNPOD_POD_ID}

expect:
  gpu_type: "NVIDIA H100 80GB HBM3"
  gpu_count: 8
  min_free_vram_gb: 70
  min_disk_gb: 200
  checkpoint_dir: /workspace/ckpt   # must be on a network volume

startup:
  deadline_s: 900
  on_failure: terminate

stall_threshold:
  default: 300
  rollout: auto
  eval: auto
  checkpoint: 900

health:
  reward_std_min: 0.01
  reward_std_patience: 20
  kl_max: 0.15
  entropy_min: 0.3
  format_success_min: 0.9
  clip_max: 0.3
  completion_len: [64, 4096]
  throughput_degradation_pct: 30

budget:
  hourly_rate_usd: 25.60
  max_usd: 400
  warn_pct: 75
  max_wall_clock_h: 24

failsafe:
  confirm_delay_s: 120
  unknown_tolerance_s: 900
  dead_mans_timeout_s: 1800
  checkpoint_timeout_s: 600
  on_terminal: terminate            # terminate | stop

notify:
  slack_webhook: ${SLACK_WEBHOOK}
  heartbeat_url: ${HEALTHCHECKS_URL}
```

---

## 8. Acceptance criteria

The library is not done when it is written. It is done when it survives injected faults. Build `tests/chaos/` with fakes for both APIs and verify each of these end-to-end:

| Injected fault | Required behaviour | Must be caught by |
|---|---|---|
| `kill -9` the trainer | Detected < 3 min; pod terminated | Sentinel |
| `SIGSTOP` the trainer (hang) | Detected within phase threshold; py-spy dump captured; pod terminated | Sentinel |
| Network cut from pod to W&B | `unknown`, not `fail`; no termination inside `unknown_tolerance`; recovers cleanly | Both |
| Fill the disk | Startup gate fails, or mid-run warn before checkpoint failure | Watchdog |
| Reward held constant (RL) | `reward_std` probe fails within `reward_std_patience` steps | Watchdog |
| Rollout duration 10× baseline | Stall detected without false-alarming on normal rollouts | Both |
| Training completes normally | `Finished`, checkpoint saved, pod terminated, spend reported | Watchdog |
| Sentinel process killed | External heartbeat service pages within its own window | External |
| Read-only API key | Startup gate fails loudly, before any GPU time is spent | Startup |

Additional bar:
- No probe may terminate on a single unconfirmed reading.
- Total sentinel cost (compute + API calls) under $1/day.
- Zero false terminations across a 24h soak on a healthy run. **A monitor that kills good runs will be switched off, and then you have no monitor at all.**

---

## 9. Wiring the live experiment

Only after §8 passes:

1. Write `configs/rlm-ft-001.yaml` from the template in §7. Measure `hourly_rate_usd` from the actual pod, don't guess.
2. Add three lines to `train.py` — gate, watchdog, shutdown wrapper. Nothing else.
3. Ensure the trainer emits `phase`, `heartbeat`, and the §5.2 metrics. Most of these already exist in TRL/verl/OpenRLHF under different key names; map them in config rather than patching the trainer.
4. Confirm `/workspace` is a network volume, so terminate is non-destructive.
5. Enable W&B's built-in "Run crashed" alert in User Settings as a free redundant channel.
6. Dry run: 50 steps, budget cap $5, verify the full ladder fires and the pod dies on its own.
7. Launch.

---

## 10. Non-goals

- Not a replacement for the W&B UI. This detects and reacts; humans still read curves.
- Not an experiment tracker, orchestrator, or scheduler.
- No auto-restart or auto-resume in v1. Automatic restarts on a misconfigured run are a way to spend money faster. Detect, halt, notify; a human decides whether to resume.
- No continuous log streaming into an agent context. Logs are pulled on alarm, bounded, for diagnosis only.