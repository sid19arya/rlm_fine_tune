# V0 smoke test — run ledger

Append-only record of what was actually run and what actually happened.
Times are UTC. Nothing here is edited after the fact; corrections are appended.

**Run identity**

| | |
|---|---|
| W&B | `rlm-runpod-1/rlm-context-management/v0-smoke` |
| rlm fork | `sid19arya/rlm@context-mgmt` (`edfe854`) |
| Budget cap | $5.00, sentinel-enforced |
| Wall-clock cap | 4h |
| Deliverable | seconds per step |

---

## Pre-flight (before any spend)

**Spend cap** — confirmed set by the human in the RunPod console. The API
exposes no billing-limit endpoint, so this is an assertion, passed to
`provision.py` as `--spend-cap-confirmed`.

**Credentials** — `RUNPOD_API_KEY`, `HF_TOKEN`, `wandb_api_key` present in
`.env`. Note the W&B key is lower-cased in the file; `provision.py` accepts
either spelling locally and always exports the canonical `WANDB_API_KEY` to the
pod, because Linux env vars are case-sensitive and the lower-case name would
simply not exist there.

**RunPod auth check**

```
GET https://rest.runpod.io/v1/pods  ->  200
existing pods: 0
```

Zero existing pods is RunPod's own connectivity pass, and also confirms nothing
is already billing.

**W&B identity check**

```
POST https://api.wandb.ai/graphql  ->  200
username:        sidthekid1
default entity:  rlm-runpod-1
teams:           ['rlm-runpod-1']
```

`rlm-runpod-1` is reachable.

**Digest reachability check** — with `rlmwatch[wandb]` installed and the real
key:

```
$ rlmwatch digest --run rlm-runpod-1/rlm-context-management/v0-smoke \
    --rate 0.88 --max-usd 5 --rollouts-per-step 32 --billing-lead-min 30
v0-smoke: cannot read the run (... Could not find run ... (not found))
```

Correct: the run does not exist yet. This is the same string Hermes will see if
it polls before launch, and it means "not started", not "broken".

---

## Timeline

<!-- appended as things happen -->

### 23:40:55Z — provision

```
$ python experiments/v0_smoke/provision.py --spend-cap-confirmed --run-id v0-smoke
checking auth and existing pods...
  auth ok, 0 existing pod(s)
  requesting 2x from NVIDIA A40 or NVIDIA RTX A6000, ~$0.88/hr
pod 81katovhvjiwu1 created.
```

**Pod `81katovhvjiwu1`.** Billing starts here, not at launch.

**A40 was not allocated — we got the A6000 fallback.** `costPerHr` came back
`0.98`, which is 2 x $0.49 (A6000), not 2 x $0.44 (A40). This is the documented
fallback working as designed: `gpuTypeIds` is a list the API treats as "any of
these", so RunPod placed on the A6000 when 2x A40 was not free. Both are Ampere
48GB, so `flash_attention_2` and the VRAM budget are unchanged. The only
consequence is cost.

Config updated: `budget.hourly_rate_usd: 0.88 -> 0.98`, measured rather than
assumed. Revised envelope:

| | |
|---|---|
| Rate | $0.98/hr |
| Wall-clock cap (4h) | $3.96 |
| $5 cap reached at | 5.10h |

Still inside budget, with less headroom than the A40 case. The wall-clock cap
binds first, as intended.

### 23:41Z — bug found in `wait_until_running`

Provisioning reported success in **4 seconds**, which is not physically
possible for a GPU pod. Cause:

```
desiredStatus: RUNNING      <- set at rental, this is the state RunPod is AIMING for
runtime:       None         <- the container is not up
publicIp:      ""           <- no address yet
```

`wait_until_running` was checking `desiredStatus`, which is RUNNING from the
instant the pod is rented. It is a *desired* state, not an observed one. The
function returned immediately and printed SSH instructions for a pod that could
not be reached.

Fixed to wait on `runtime` being populated, which is when the container is
actually up and ports are mapped. Worth noting for anyone reading the RunPod
API: `desiredStatus` is not a readiness signal.

Now polling for real readiness.

### 23:48Z — pod 1 was unreachable: no SSH key

`publicIp` and `portMappings` populated at ~T+8min, but `runtime` stayed
`None` throughout — so `runtime` is *also* not a reliable readiness signal.
SSH reached the daemon and was refused:

```
$ ssh root@194.68.245.232 -p 22138
root@194.68.245.232: Permission denied (publickey,password).
```

No keypair had ever been registered. RunPod injects `authorized_keys` from the
`PUBLIC_KEY` env var **at container start**, so it cannot be added to a pod that
is already running. Pod 1 was unusable and was terminated.

```
$ 23:50:11Z terminate 81katovhvjiwu1  ->  pods now: 0
```

Cost of the lesson: ~9 min at $0.98/hr ≈ **$0.15**.

Two fixes to `provision.py`:
- generate/inject `PUBLIC_KEY` from `~/.ssh/runpod_rlm.pub`, and refuse to
  provision at all if no key exists — better than paying to boot an
  unreachable box.
- readiness now waits on `publicIp` **and** a mapped port 22, having learned
  that neither `desiredStatus` nor `runtime` means reachable.

### 23:50:19Z — pod 2

```
pod kri19pywwj9kzt created
ssh root@194.68.245.3 -p 22010
```

Provisioning still crashed on its success path: `portMappings` is a flat
`{"22": 22010}` map, not a list of `{privatePort, publicPort}` objects. Fixed.
The pod itself was fine.

**The GPUs are A40 after all.**

```
0, NVIDIA A40, 46068 MiB, 45489 MiB
1, NVIDIA A40, 46068 MiB, 45489 MiB
```

So the $0.98/hr is A40 priced at $0.49/GPU in this datacenter, not the $0.44
list figure — the earlier "A6000 fallback" reading was wrong, it was A40 all
along at a higher regional price. Hardware is exactly what the experiment
specified. `hourly_rate_usd: 0.98` stands, being measured.

`/workspace` is an mfs network mount (`mfs#eu-se-1.runpod.net:9421`). Note
this is RunPod's backing store for the volume disk, **not** a network volume in
the billing sense — it is still destroyed on terminate, so the
upload-before-terminate rule is unchanged.

### 23:51Z — env vars do not reach SSH sessions

Everything passed at creation is in PID 1's environment:

```
WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_RESUME WANDB_RUN_ID PUBLIC_KEY
```

but `/etc/rp_environment`, which `~/.bashrc` sources, holds only RunPod's own
`RUNPOD_*` vars. An SSH session — and every tmux pane, being another fresh
shell — sees none of them.

Unfixed, the trainer would have found no `WANDB_ENTITY`/`PROJECT`/`RUN_ID`, and
W&B would have created a random run id in the default entity. That is precisely
the silent failure the pinned run identity was meant to prevent, arriving by a
different route: the monitor and Hermes would both have polled a path that
never existed.

`setup.sh` now copies them out of `/proc/1/environ` into `/etc/rp_pod_env`,
sources it, and appends it to `~/.bashrc` so every later shell inherits them.
It hard-fails if `WANDB_RUN_ID` is still missing afterwards.

### 23:53Z — setup, attempt 2: env quoting

Died in 3 seconds:

```
/etc/rp_pod_env: line 6: export: `rlm-v0-smoke': not a valid identifier
```

`PUBLIC_KEY` is `ssh-ed25519 AAAA... rlm-v0-smoke` — three space-separated
words — so an unquoted `export` split on them. Values are now written with
`printf %q`, and `PUBLIC_KEY` is excluded outright (RunPod's start script
consumes it; nothing downstream needs it). `set -e` caught this before anything
else ran, which is the cheapest possible way to find it.

Quoting was verified locally against a synthetic `/proc/1/environ` before
spending another pod-minute on it.

### 23:53:13Z — setup, attempt 3: everything but the model

Env import worked:

```
imported: HF_HOME HF_TOKEN WANDB_RESUME WANDB_PROJECT WANDB_ENTITY WANDB_RUN_ID WANDB_API_KEY
```

All green through: hardware check (2x A40, 45489 MiB free each), volume disk,
HF cache pinned to `/workspace/hf`, uv, the pinned rlm fork, prime-rl clone,
`rlmwatch==0.1.0` installed from `/workspace/rlm_fine_tune`, and
`training/configs/smoke.toml` written.

Then, at the last step, `SETUP_EXIT=127`:

```
=== pre-downloading Qwen3-8B ===
/root/setup.sh: line 132: huggingface-cli: command not found
```

`huggingface-cli` is not pulled in by any of rlm's dependencies, and recent
`huggingface_hub` renamed the command to `hf` — so invoking either by name is a
coin flip. Replaced with `snapshot_download` from the Python API, which is the
same code path and stable across versions, plus an explicit
`huggingface_hub>=0.24` install.

Note the ordering worked in our favour: the download is deliberately the last
step, so this failed *after* all the expensive setup had succeeded and the
re-run skips straight past it (the script is idempotent).

### 23:55Z — setup, attempt 4: the 16GB download

Re-running. This is the long step.

### 23:56Z — attempt 5: `uv venv` is not idempotent

```
error: Failed to create virtual environment
  Caused by: A virtual environment already exists at: .venv
```

The script advertised itself as re-runnable and was not. Changed to
`uv venv --python 3.12 --allow-existing` — deliberately *not* `--clear`, which
would discard installs that had already succeeded on an earlier attempt.

Confirmed in the same run: `rlm pinned at edfe854`, the `enable_sub_lm` commit.

### 23:56Z — attempt 6: CRLF, self-inflicted

```
/root/setup.sh: line 11: set: pipefail: invalid option name
```

`pipefail\r` is not an option name. The file had acquired CRLF line endings —
**my own doing**: I had rewritten `setup.sh` with Python's `write_text()`, which
on Windows translates `\n` to `\r\n`. Attempts 1-4 ran fine because those edits
went through tools that did not translate.

Two fixes:
- the file is now rewritten via `read_bytes`/`write_bytes`, which does no
  newline translation;
- the transfer pipes through `sed 's/\r$//'` on the way to the pod, so a CRLF
  working copy can never produce this again regardless of how it got that way.

Worth recording as a class: three of the six setup failures were introduced by
the tooling around the script rather than by the pod. A Windows control machine
driving a Linux pod has this hazard everywhere, and the cheap defence is to
normalise at the boundary rather than trust the working copy.

### 00:17Z — setup complete, and the model landed fast

`SETUP_EXIT=0`. The download the spec budgeted ~25 minutes for took **13
seconds**:

```
Fetching 15 files: 100%|██████████| 15/15 [00:13<00:00,  1.19s/it]
model at /workspace/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d... (15.3 GiB)
```

~1.2 GB/s on RunPod EU to the HF CDN. Not taken at face value — the snapshot
directory holds 76-byte symlinks into `blobs/`, so sizes were re-read
dereferenced:

```
model-00001-of-00005.safetensors  3.72 GiB
model-00002-of-00005.safetensors  3.72 GiB
model-00003-of-00005.safetensors  3.69 GiB
model-00004-of-00005.safetensors  2.97 GiB
model-00005-of-00005.safetensors  1.16 GiB
arch: ['Qwen3ForCausalLM'] | layers: 36 | hidden: 4096
```

On `/workspace`, not the 30GB container disk. For V1 planning: the download is
not a meaningful cost in this datacenter, and `startup.deadline_s: 3600` is very
conservative.

### 00:17Z — the budget cap was inert (most serious defect so far)

Checking spend returned `$0.00` on a pod that had been running half an hour.

```
PodStatus.uptime_s = 0.0  -> spend would be $0.00
```

The live API populates **neither** `uptimeSeconds` nor `runtime` on a running
pod. Every cost probe computes spend as `uptime x rate`, so a permanent zero
means:

- `cost.spend` never warns and never fails,
- `budget.max_usd` — the $5 cap — can never be reached,
- `cost.cost_per_progress` reports nothing,
- and the digest tells Hermes the run is free.

The single safeguard that has to work without a human present would have been
silently switched off for the whole run, while reporting healthy.

`createdAt` is present and is the correct clock anyway, since billing starts at
provisioning rather than at boot. The client now falls back to it. RunPod
returns a Go-style stamp (`2026-09-07 23:50:20.758 +0000 UTC`) which is not ISO
8601 and which `fromisoformat` rejects, so it is parsed explicitly.

Verified against the live pod:

```
uptime 30.2 min -> spend $0.49 of $5.00
```

Six tests cover it, including that unparseable input yields 0.0 rather than an
exception.

### 00:20-00:35Z — prime-rl: three layered failures

`setup.sh` deliberately leaves prime-rl's install to a human, since its README
changes often. Doing it turned up three problems in sequence.

**1. Submodules were never initialised.** A plain `git clone` leaves `deps/`
empty, and `uv sync` died with

```
× Failed to build `prime-pydantic-config @ file:///workspace/prime-rl/deps/pydantic-config`
╰─▶ ... does not appear to be a Python project
```

**2. Four of the five submodules use `git@github.com:` SSH URLs.** There is no
GitHub key on the pod and there should not be one. A *local*
`url.<base>.insteadOf` did not reach the submodule clone subprocesses — they
still tried SSH and died on host-key verification. Fixed with a `--global`
rewrite plus editing `.gitmodules` directly and `git submodule sync`.

**3. `--depth 1` left the working trees empty.** `.git` present, correct commit
checked out, every file staged as deleted. Restored with

```
git submodule foreach --recursive 'git checkout -- . || git reset --hard -q HEAD'
```

Then `uv sync --all-extras` succeeded: 303 packages, flash-attn-4 built from
source, `SYNC_EXIT=0`.

### 00:31Z — the monitor was lying, and I wrote the lie

`PROBE-FAIL: pod unreachable` fired three times against a pod that was fine
every time. Cause: `ps aux | grep -c` **exits 1 when the count is zero**, so
"no python processes running" was reported as "pod unreachable".

This is the same defect class the whole library exists to catch — a monitor
reporting a confident wrong thing — and it was in the monitoring I wrote for
this session. Fixed with `|| true`, and the two states are now distinguished:
`UNREACHABLE` only when ssh returns nothing at all, `IDLE` when it returns real
numbers showing no activity.

The rebuilt monitor immediately proved useful:

```
MOVING net=51104KB/s disk=103601KB/s gpu=0% procs=1
```

Throughput, not log-line presence. During the model download I had only a log
grep, which cannot distinguish a stalled transfer from a quiet one.

### 00:33Z — container disk at 50%, uv cache in the wrong place

```
overlay      30G   15G   16G  50% /
/root/.cache 15G                     <- uv cache, on the CONTAINER disk
```

`HF_HOME` was pinned to `/workspace`; `UV_CACHE_DIR` was not. Same "no space
left on device" failure, different cache — prime-rl pulls torch, vLLM and ~300
other packages. It fitted this time with 16G to spare, but only just.

`setup.sh` now pins `UV_CACHE_DIR=/workspace/uv-cache` and sets
`UV_LINK_MODE=hardlink`, which also removes the "Failed to hardlink files,
falling back to full copy" warning and the doubled disk write behind it.

### 00:36Z — the verifiers conflict resolved itself

Installing rlm's packages into prime-rl's venv replaced three of its vendored
submodule packages with PyPI versions:

```
- verifiers==0.0.1.dev1 (from deps/verifiers)  ->  + verifiers==0.3.1
- renderers==0.0.1.dev1 (from deps/renderers)  ->  + renderers==0.1.11
- mcp==2.0.0                                    ->  + mcp==1.29.1
```

prime-rl pins those as local submodules deliberately; `rlm/training` declares
`verifiers>=0.1.11`, which resolves from PyPI and clobbers the vendored fork.
This was the first thing in the run with a real chance of being unfixable in
minutes. It was not:

```
verifiers environments: ['oolong']
prime_rl, rlm_train, oolong all import
```

prime-rl tolerates upstream verifiers 0.3.1, and `oolong` registers as a
`verifiers.environments` entry point, which is how `rl` discovers it. `uv pip
install --inexact` was used throughout, since prime-rl's README warns a plain
`uv sync`/`uv run` will uninstall anything outside its lockfile — which would
have silently removed the packages being added.

### 00:38Z — GATE: ablation arm verified on the live pod

```
$ verify_ablation.py --rlm-root /workspace/rlm --config .../smoke.toml
verifying the NO recursion (ablation) arm against /workspace/rlm

  [ok] prompt names no sub-LM function
  [ok] prompt describes no delegation
  [ok] REPL binds no sub-LM tool
  [ok] _restore_scaffold does not re-bind the tools
  [ok] prompt and REPL agree on every tool name
  [ok] rubric does not gate on sub-calls

The NO recursion (ablation) arm is correctly selected.   exit 0
```

Verified against the actually-installed code rather than a local simulation.
The `_restore_scaffold` line is the one that matters most: it re-installs the
REPL scaffolding after every turn, so a flag honoured only at setup would have
restored delegation on turn two, mid-rollout, invisibly.

`rl --help` responds, so prime-rl's entry point is live.

### 00:40Z — GATE FAILED: prime-rl config schema mismatch

Launched under tmux; the run died immediately on config validation:

```
7 validation errors for RLConfig
--orchestrator.train.env            Extra inputs are not permitted
--orchestrator.rollouts-per-example Extra inputs are not permitted (got 4)
--orchestrator.filters              Extra inputs are not permitted
--inference.gpu-memory-utilization  Extra inputs are not permitted (got 0.8)
--inference.model                   Extra inputs are not permitted -- did you mean --model?
--inference.parallel                Extra inputs are not permitted
--wandb                             Extra inputs are not permitted
```

Not typos. prime-rl's schema moved in `d135ed4c9` (2026-07-29),
*"rename env collections to sources, compose the verifiers config blocks"*:
`[[orchestrator.train.env]]` with `id`/`args` became
`[[orchestrator.train.source]]` with `env.taskset.id`.

Version archaeology:

| | commit | date |
|---|---|---|
| rlm fork | `854e688` | 2026-08-25 |
| prime-rl installed | `04a61d3b7` | 2026-09-07 |
| breaking rename | `d135ed4c9` | 2026-07-29 |

rlm pins no prime-rl version anywhere, and its example config uses
pre-rename schema despite postdating the rename — so the harness targets an
older prime-rl than HEAD.

**A correction to an earlier entry.** The "entry point discoverable" check that
passed at 00:36 verified `oolong` registers under `verifiers.environments` —
but HEAD prime-rl's config never reads that group. The check passed while
testing a mechanism this version does not use. It was weaker evidence than it
appeared, and the config validation is what actually caught the mismatch.

### 00:47Z — pinning prime-rl to 1fd2d732c

Pulled the config from the commit immediately before the rename. It matches
`smoke.toml` almost exactly — `[deployment] num_train_gpus`, `[wandb]`,
`[trainer.model.lora]`, `[[orchestrator.train.env]]` all present and correctly
spelled. Strong evidence the config was right for the wrong version rather than
simply wrong.

On scope: pinning a dependency is repair, not redefinition. The spec's "may
not" list is *changing max_steps, batch size or model to make it work*, and
*install packages, edit config files* is in the "may" column. Rewriting the
config into HEAD's `source`/`taskset` schema would be redefinition — inferring
semantics with no rlm example to copy — and that remains a human decision.
Those two were bundled in the first escalation; they should not have been.

First attempt aborted in seconds:

```
error: Your local changes to the following files would be overwritten by checkout:
	.gitmodules
```

— my own earlier HTTPS rewrite blocking the checkout. `set -e` stopped it
before `rm -rf .venv`, so the working environment was never damaged.

### 00:52Z — the pin does not fix it either. Two failure modes, one root cause.

Pinning to `1fd2d732c` was based on that commit's config matching `smoke.toml`.
Checking the *env block* specifically, before launching, showed it does not:

```toml
# 1fd2d732c
[[orchestrator.train.env]]
name = "alphabet-sort"
env.taskset = { id = "alphabet-sort-v1", ... }
env.agent.harness = { id = "null" }

# what rlm's config needs
[[orchestrator.train.env]]
id = "oolong"
[orchestrator.train.env.args]
dataset_name = "spam"
```

The `taskset`/`agent` structure **predates** the rename I had identified. Last
prime-rl configs using `env.args`: **2026-07-20**. `group_size` vs
`rollouts_per_example` confirms the same drift independently.

Running it anyway gave a second, different failure:

```
AttributeError: module 'verifiers.v1' has no attribute 'EnvServerConfig'
```

Pinned prime-rl needs its **vendored** verifiers (`deps/verifiers` @ `d0bb0ff`);
`rlm-train` declares `verifiers>=0.1.11`, which resolves from PyPI and has an
incompatible API. They cannot coexist in one environment. (An earlier
`connect-python` clash was cleared using the remedy its own error named — that
one was genuinely fixable.)

**Root cause, stated plainly:** `alexzhang13/rlm`'s training harness targets a
prime-rl from ~mid-July 2026 and pins no version — not in `pyproject.toml`, not
in a lockfile, not in the README. prime-rl has since made at least two breaking
changes. Anyone following rlm's training README today gets a config that cannot
validate against any prime-rl they would plausibly install.

### 00:56:30Z — stopped, not terminated

```
before: RUNNING, uptime 66 min, spend $1.08
stop sent
after : EXITED
```

`stop` rather than `terminate`: compute billing halts, the volume disk survives
at ~$0.01/hr, and the 15.3 GiB model plus both checkouts are still there if a
correct prime-rl commit turns up. Terminate destroys all of it and there is no
network volume. This was the reversible option and the agent took it without
waiting, because the alternative was idling at $0.98/hr for an unbounded period.

**Total spend: ~$1.23** (pod 1 $0.15 + pod 2 $1.08) of a $5 cap.

---

## V0 outcome

**The deliverable was not obtained.** No seconds-per-step, because not one
training step ran.

**What was proven to work:** provisioning, SSH with an injected key, env
propagation into non-interactive shells, 2x A40 45.5GB each, Qwen3-8B 15.3 GiB
verified on the volume disk, prime-rl building from source including
flash-attn-4, `rlmwatch` installed and running on the pod, and the ablation arm
verified green against live code — including the `_restore_scaffold` turn-two
trap that a source-scanning approach could never have caught.

**What blocked it:** rlm and prime-rl have diverged and rlm pins no version.

**What the run found in rlmwatch itself:** the budget cap was inert, because
RunPod populates neither `uptimeSeconds` nor `runtime` on a live pod and spend
therefore computed as $0.00 forever. That is the exact failure class the library
exists to catch, and only a live pod surfaced it.

**11 interventions logged**, 2 unrecoverable — both the same root cause.

### 01:05Z — the compatible prime-rl commit, found by dating the harness

The right method is not to ask upstream, it is to date the harness and diff
schemas. `rlm/training/configs/` has **one commit in its entire history**:

```
de762b9  2026-05-24  add training harness and update the local REPL slightly
```

So the harness targets prime-rl from late May 2026 — not mid-July, as estimated
earlier from a rename commit message. That estimate was wrong by seven weeks,
and the two failed pins followed from it. Dating the config should have come
first.

**Candidate: `083127fe` (2026-05-27), three days after the harness landed.**
Every key prime-rl HEAD rejected exists there, read from the schema rather than
inferred:

| key | evidence at `083127fe` |
|---|---|
| `[[orchestrator.train.env]]` `id`/`args` | `EnvConfig`: `id: str`, `name`, `args: dict` |
| `rollouts_per_example` | `AliasChoices("group_size", "rollouts_per_example")` |
| `[[orchestrator.filters]]` | `RepetitionFilterConfig`, `ZeroAdvantageFilterConfig` |
| `[inference] gpu_memory_utilization` | `gpu_memory_utilization: float = 0.9` |
| `[inference.parallel]` | `ParallelConfig` (tp/dp) |

`rollouts_per_example` surviving only as an *alias* is the convincing detail:
rlm's author wrote against a prime-rl where it still resolved.

**Two risks git cannot settle.** May-2026 prime-rl pins older torch/vLLM, and
whether that stack builds on an A40 today is unknown until tried. And its
vendored `verifiers` may still clash with `rlm-train`'s `verifiers>=0.1.11` —
that conflict is orthogonal to the config schema and could still bite.

### Correction: "verifiers without prime-rl" was vaguer than it should have been

Stated earlier as an alternative without qualification. Precisely:

- **verifiers** is the environment/rollout framework; `oolong` registers into it
  via the `verifiers.environments` entry point.
- **prime-rl** is the training loop — trainer, weight sync, GRPO.

verifiers alone can load the env, run rollouts against a vLLM server, execute
the REPL and score with the rubric. It **cannot train**: no optimizer, no weight
updates, no policy gradient.

So it does **not** produce V0's deliverable. Seconds-per-step measures a
training step and there would be none. What it produces is base-model
performance through the REPL harness — useful later as cells A0/A of V1's eval
protocol, but not V0, and it should not have been offered as an alternative
without saying so.

---

## Session 2 (03:43-04:56Z) — the pinned commit works; the run does not

### 03:43Z — the stopped pod could not restart

```
POST /pods/kri19pywwj9kzt/start -> 500
{"error":"start pod: There are not enough free GPUs on the host machine"}
```

The A40s were reallocated during the 2.8h pause. This is the risk named when
`stop` was recommended, and it landed. Terminated and provisioned fresh
(`pqh54v1mcyare2`). Stopping was still correct: 2.8h running would have been
~$2.75 against ~$0.03 stopped.

Setup on the fresh pod passed **first time in ~4 minutes**, against six attempts
and ~40 minutes on the first pod. The fixes paid for themselves.

### 04:02Z — GATE PASSED: the config validates at 083127fe

The archaeology was right. Staged deliberately so the three risks could be told
apart:

```
Resolved 452 packages, 384 installed, flash-attn-4 + transformers built  <- deps OK
rl entrypoint imports OK          <- prime-rl alone (previous pin died here)
still OK                          <- survived rlm's deps; no verifiers clash
outputs/rlm-v0-smoke created      <- prime-rl READ output_dir from smoke.toml
Starting inference on GPU(s) 0 / trainer on GPU(s) 1 / SUCCESS Startup complete
```

Every key HEAD rejected was accepted. And the experimental condition reached the
live process:

```
enable_sub_lm: False   min_subcall: 0   enable_thinking: False
dataset_name: 'spam'   max_completion_tokens: 2048
vLLM: max_model_len 16384, enable_lora, max_lora_rank 32
trainer: attn='flash_attention_2', LoRA 30,670,848 params on 1,509,949,440 base
```

### 04:11Z — the pinned run id was ignored

`WANDB_RUN_ID=v0-smoke` was set precisely to make the path predictable. prime-rl
uses wandb **shared mode** and generated `492e1de2ed394ae68e6ec12f34d654a5`;
`v0-smoke` survived only as the display name. Both the monitor config and the
Hermes briefing were pointing at a 404. Caught by checking both paths against
the API rather than assuming the env var won.

Lesson for V1: **read the run id out of the launch log; do not try to set it.**

### 04:22-04:38Z — three self-inflicted failures in a row

1. **Filled the volume.** `UV_CACHE_DIR` on `/workspace` took the 100GB volume
   to 91GB (two dependency sets). Training died on `Disk quota exceeded` mid
   weight-conversion, no traceback, whole process group gone.
2. **`rm -rf` deleted the environment.** uv 0.12 stores environments *inside*
   the cache dir; `.venv` was a symlink into it.
3. **`uv cache prune` deleted it again.** 82532 files removed for 10.6MiB --
   hardlink targets that environment files pointed into.

`UV_CACHE_DIR` is not a disposable cache on uv 0.12. Neither `rm -rf` nor
`prune` is safe against it.

### 04:54Z — BLOCKED: silent death, cause not found

Four launches, all dying during startup. Every hypothesis tested and excluded:

| Suspected | Evidence |
|---|---|
| Disk quota | writes OK, no quota error, 77G of 100G |
| cgroup OOM | limit 103G, peak 50G, `oom_kill 0` |
| Host memory | 56G used of 503G |
| Container disk | 258M of 30G |
| Python exception | no traceback in any of five logs |
| tmux / session reaping | **canary test below** |

The canary settled the last one:

```
canary (sleep 3600 in tmux, started 04:52:25):  ALIVE
training (setsid, outside tmux):                DEAD
train4.log:                                      0 bytes
```

tmux is not being reaped; a process started at the same instant survives. The
training dies on its own, writing nothing. Consistent with a native-level crash
-- segfault or a CUDA/driver abort -- that produces no Python output.

Stopped rather than escalate to py-spy/gdb against a process that dies in under
two minutes writing zero bytes. Pod stopped (not terminated): `$1.13` on this
pod, **~$2.36 total** of a $5 cap.

---

## Session 3 — the deaths were not undiagnosable; the launcher destroyed the evidence

### 05:38Z — RunPod keeps nothing after a stop

Queried container logs for `pqh54v1mcyare2` via REST, hoping the runtime had
captured what `train4.log` did not. All three sources empty:

```
state=EXITED  uptime=4988s  rate=$0.98/hr
logs source=None    -> (empty)
logs source=stderr  -> (empty)
logs source=system  -> (empty)
```

Status survives a stop; **logs do not**. Post-mortem has to happen while the pod
is up, or be written to the volume. Recorded as unrecoverable.

### 05:40Z — the real finding

"Consistent with a native crash" was an inference, and I stopped at it. Three
standard diagnostics for a process that dies without output were never run, and
the launcher I used made two of them impossible:

```
setsid nohup uv run rl --config smoke.toml > train.log 2>&1 &
```

| Defect | Consequence |
|---|---|
| `setsid ... &`, never reaped | exit status discarded 4x. 139/134/137 would have named the cause outright |
| stdout block-buffered to a file | up to 8KB of startup output lost at signal death |
| no `PYTHONFAULTHANDLER` | native crash produces no stack, so logs look clean |

**`train4.log` being 0 bytes does not mean the process wrote nothing.** Python
block-buffers stdout in 8KB chunks when it is a file, and a signal death never
flushes. It means it died before filling one buffer. I read a property of my
own redirection as a property of the trainer, and built the "writes nothing"
conclusion on top of it.

Note the launcher was never a committed artifact -- it was typed into an SSH
session four times. Nothing that decides how a run dies should live only in
shell history.

### Fixed: `experiments/v0_smoke/launch.sh`

Records the exit code to a file, sets `PYTHONFAULTHANDLER=1` and
`PYTHONUNBUFFERED=1`, sets `CUDA_LAUNCH_BLOCKING=1` for diagnostic runs, names
the signal (134 SIGABRT / 137 SIGKILL / 139 SIGSEGV), and dumps `dmesg` after
death. Deliberately `set -uo pipefail` without `-e`: the point is to observe a
failing command, not to abort on it.

Cost to re-run with this in place: **~15 minutes, ~$0.25** of the $2.64
remaining under the $5 cap.

### 05:55Z — the old pod is unrecoverable; A40 stock is gone account-wide

Five `start` attempts, 45s apart, all identical:

```
500 {"error":"start pod: There are not enough free GPUs on the host machine..."}
```

Provisioning a replacement then failed too:

```
500 {"error":"create pod: There are no instances currently available"}
```

Not a host quirk -- a shortage. RunPod REST v1 has no GPU catalogue endpoint,
but the GraphQL API does, and it is a free read:

```
secure    : A40 and A6000 ABSENT. Cheapest 2x >=40GB is L40S @ $2.18/hr
community : L40S 2x @ $1.58/hr, RTX PRO 5000 @ $1.64, A100 SXM @ $2.78
```

The pod we were using at $0.88-0.98/hr is not purchasable right now at any
tier. **Worth adding to the provisioner: query GraphQL stock before attempting
creation, so a stock-out is a pre-flight message rather than a 500.**

Terminated `pqh54v1mcyare2` (204, 0 pods remain) -- the guard in provision.py
correctly refused to create a second pod while a stopped one was still billing
storage, and the stopped pod's volume was unreachable anyway.

### 05:58Z — third bug in the cost path, found by reading the pod list

`createdAt`-based uptime (the fix that made the budget cap work at all) had no
state check, so an EXITED pod kept accruing wall-clock: `pqh54v1mcyare2`
reported a rising uptime against $0.98/hr while stopped. Fail-safe in
direction -- spend is over-reported, so the cap trips early -- but it would
strand a budget on a pod that stopped spending hours ago. Now returns 0 uptime
for any non-RUNNING pod. 348 tests pass.

That is three defects in one small function, all found by running it against
live infrastructure and none reachable with fakes.

---

## Session 3 (cont.) — the fresh pod, and what setup.sh had silently lost

### 15:39Z — stock returned; pod `gfbjns3cqfa9cv`

A GraphQL stock poller (free read, 5-min interval) fired at 15:39 and
auto-provisioned. 2x A40 @ $0.98/hr, `194.68.245.29:22167`. Nothing billed
during the ~4h wait -- 0 pods.

### 15:42Z — setup.sh finished in 90 seconds. That was the bug.

Last session setup took ~25 minutes. This time: 90s. The model pull was
genuinely fast (measured 119 MB/s off `/proc/net/dev`, 14GB, 15.3 GiB verified
on disk) but the real reason was that **setup.sh does not build prime-rl at
all** -- it cloned the repo and printed "follow prime-rl's README, then re-run".

So the single most valuable result of the previous session -- the pin to
`083127fe`, found by dating `rlm/training/configs/` -- **lived only in my shell
history and died with the pod.** The fresh pod came up on HEAD (`04a61d3b7`).

Fixed in `2dde535`. setup.sh now pins, inits submodules and builds.

### 15:43-15:47Z — three named failures in eleven minutes

The instrumented launcher earned itself here. Against four prior launches that
produced nothing:

| exit | cause | fix |
|---|---|---|
| 127 | `stdbuf: failed to run command 'uv'` | uv is in ~/.local/bin, added by ~/.profile, which a detached shell never reads (`492b93d`) |
| 1 | `prime-pydantic-config does not appear to be a Python project` | submodules never initialised; several use `git@github.com:` SSH URLs and the pod has no key |
| - | `Failed to clone 'configs/private' a second time, aborting` | at this commit the submodule set includes a PRIVATE repo, and git aborts the WHOLE update on it. Scope to `-- deps` |

**One root cause, three faces: environment that exists only in an interactive
shell is invisible to a detached process.** PATH, `UV_CACHE_DIR` (set only in
~/.bashrc; a detached `uv sync` would have put a 15GB cache back on the 30GB
container disk), and the prime-rl pin itself.

### 15:51-16:00Z — 20 minutes lost to my own shell precedence bug

```
ssh '... cat > f && chmod +x f && setsid nohup bash f > log 2>&1 &  sleep 2; echo LAUNCHED'
```

`&` backgrounds the ENTIRE `&&` chain, so `cat` was still reading the script
from ssh's stdin when `echo LAUNCHED` returned and ssh tore the connection
down. Result: a **0-byte script**, a **0-byte log**, and a process that
appeared to start and vanish.

That is precisely the signature I spent last session reading as a native crash.
Here it was a botched file transfer. It does not overturn the earlier reading
-- those runs demonstrably printed LoRA parameter counts, so the binary was
real -- but twice now I have read "0 bytes" as a fact about the trainer when it
was a fact about my plumbing.

My liveness check was also lying: `pgrep -f bootstrap2.sh` was matching my own
monitor's ssh command line. Now `pgrep -c -f "bash /root/bootstrap2.sh"`.

### 16:01-16:12Z — build, and two monitors that lied by omission

`uv sync` in ~8 min (vs ~20 last session). Then:

- **Watcher v1 emitted nothing when the log stayed empty** -- a script that
  never started looked identical to quiet progress. Exactly the failure mode the
  tooling docs warn about. Fixed by treating a flat log as an event.
- **Watcher v2 mis-parsed** `B=292 P=3 C=3835`: a greedy `sed 's/.*P=\([0-9]*\)/'`
  captured the trailing fields too. Rewritten to emit three plain lines read
  positionally.

Also learned: **this shell collapses `\` inside heredocs**, so every escaped
regex written through one arrives mangled -- `\1` reached Python as `\1`,
an octal escape for chr(1). It corrupted a setup.sh patch and both watcher
fixes. Stopped writing escaped regex through heredocs entirely.

`uv sync` needs a real liveness signal because `| tail -20` withholds all
output until the pipeline ends. Cache growth is that signal: 3.8 -> 13.3 ->
25.1 -> 32.0 GB.

Build verified: `sync rc=0`, all four deps populated, `verifiers environments:
['oolong']`, `prime_rl + rlm_train import OK`. Volume 46GB of 100GB quota,
container disk 193M of 30G.

### 16:12Z — instrumented trainer launched

`/workspace/runs/20260908T161203Z/`. Budget watchdog raised from an arbitrary
$1.50 to **$2.10** cumulative on this pod -- sized to let a full 20-step run
FINISH rather than to end it early, keeping the $5 account cap with ~$0.55
margin. Stopping a working run at step 3 to save $1 would have been the wrong
trade, and the first cap was set without doing that arithmetic.

### 16:25-16:31Z — run 5 reached training and died on the volume quota

Furthest yet. The ablation arm confirmed live in the real process:

```
oolong-spam-train args={'dataset_name':'spam','enable_sub_lm':False,'min_subcall':0,
                        'min_ctx':32768,'max_ctx':65536,'max_iterations':12}
sampling: temperature=1.0, max_completion_tokens=2048, enable_thinking=False
vLLM: Qwen3-8B, max_model_len 16384, enable_lora, max_lora_rank 32, gpu_util 0.8
W&B run: 4d3d911c450c433ba8dc7d4157b4c321   (shared mode again; NOT v0-smoke-diag)
```

Then, in `trainer.log` (not train.log -- the top-level log showed only
"Orchestrator failed", the cause was one level down):

```
safetensors_rust.SafetensorError: Error while serializing:
I/O error: Disk quota exceeded (os error 122)
```

### The 100GB volume was never big enough

`du` said 70.7GB of 100GB -- 29GB apparently free. An empirical write probe
said otherwise:

```
5GB  write: OK
10GB write: FAILED
```

**The volume charges quota at roughly 1.33x raw bytes.** 70.7GB real is ~94GB
charged, leaving ~6GB -- which is exactly what the probe found. A 100GB volume
yields ~75GB usable.

Measured requirement:

| item | GB |
|---|---|
| uv environment (torch, vLLM, ~300 pkgs) | 33 |
| Qwen3-8B in HF cache | 20 |
| oolong-synth dataset | 11 |
| weight broadcast, 8B bf16 via filesystem | ~16 |
| **total** | **~80-90** |

So the smoke test needs ~85-90GB against ~75GB available. It could not have
fitted. Last session's identical "Disk quota exceeded" was blamed on
`UV_CACHE_DIR`; the cache made it worse but was never the whole story.

`provision.py` now specifies **200GB**, with the measured budget recorded.

**Never trust statvfs on this volume.** `df` reports the MooseFS cluster (191T
free) rather than the quota. The only reliable check is to write a probe file.

Pod `gfbjns3cqfa9cv` terminated (204, 0 pods). Spend this pod ~$0.87;
running total ~$3.23 of $5.

### 17:29-17:50Z — ROOT CAUSE: the env worker is killed every 30 seconds

Run 6 reached real training on the 200GB pod:

```
Student inference pool ready
Initializing weight broadcast (type='filesystem')
Starting orchestrator loop (max_steps=20)
Starting orchestrator step 0
Generating rollouts (train): 0/8
```

Then nothing for 13 minutes. Both GPUs at 0%, 3 completion requests total,
CPU workers churning. The tell was that `rlm_train.worker` PIDs changed between
two samples 25s apart -- not hung, **respawning**.

`logs/envs/train/oolong-spam-train/env_server.log` had it:

```
EnvRouter - WARNING - Worker 0 heartbeat timeout (32.8s), restarting
EnvRouter - INFO - Started worker (id=0, name=oolong-0, pid=13535)
...
Active tasks: 0 (W0: ?)
```

`worker_heartbeat_timeout: float = 30.0` in
`verifiers/serve/server/env_{server,router}.py`. Building a 32768-65536 token
oolong context takes longer than 30s, so the router kills its own worker
roughly every 66 seconds, forever. No rollout can ever complete. It is not
exposed through smoke.toml.

**This is very likely what last session's "silent hang" actually was.** The
work was being killed faster than it could finish, and with block-buffered
stdout none of it was visible.

A second defect compounds it, in rlm's own oolong environment:

```
prompt_messages returned raw dicts/strings instead of vf.Messages.
This repeatedly triggers normalize_messages
```

Wrong return type forcing repeated re-normalisation -- slowing the exact step
that blows the heartbeat. Worth fixing in the fork regardless.

Raised the timeout to 900s on the pod and relaunched (run 7).

### Diagnostic notes

* The error was in a THIRD log level. `train.log` said only "Orchestrator
  failed"; `trainer.log` had the quota error; the heartbeat kill was in
  `logs/envs/train/<env>/env_server.log`. Four log levels, and the cause is
  never in the top one.
* `py-spy` cannot attach here -- the container lacks CAP_SYS_PTRACE
  ("Failed to copy Py_Version symbol: Permission denied"). Comparing PIDs and
  `/proc/<pid>/stat` cpu ticks across samples worked instead, and distinguishes
  respawn from hang without any privileges.
* `pkill -9 -f vllm` matched my own ssh command string and killed the session.
  Kill by PID from `nvidia-smi --query-compute-apps` instead.

### 18:03Z — THE DELIVERABLE: 420.66 seconds per step

```
SUCCESS Step 0 | Time: 420.66s | Reward: 0.1250
               | Seq. Length: 6857.1 tokens/sample | Max. Off-Policy Level: 0
```

Step 0 ran 17:56:06 -> 18:03:15. **420.66s/step on 2x RTX A6000 @ $1.06/hr.**

| | wall time | cost |
|---|---|---|
| 20-step smoke | 2.3h | ~$2.48 |
| **250-step real run** | **29.2h** | **~$31** |

**The real run is affordable.** That is the question V0 existed to answer.

### What the heartbeat fix changed

| | run 6 (30s timeout) | run 7 (900s) |
|---|---|---|
| Active tasks | 0, always | 8 concurrent |
| GPU 0 utilisation | 0% | 99-100% sustained |
| Worker kills | every ~66s, unbounded | 0 after relaunch |
| Steps completed | 0 | 1, in 420s |

### Two caveats that matter more than the headline

1. **`Detected 4/8 rollouts (zero_advantage=4), enforced 4`.** Half the batch
   produced no learning signal. With 2 more lost to `WorkerStartupError`, the
   effective batch was ~2 of 8. The wall-clock estimate holds; the sample
   efficiency does not. Fix before committing 29 hours.
2. **`Seq. Length: 6857.1 tokens/sample`** against a configured `min_ctx=32768,
   max_ctx=65536`. Rollouts are terminating far earlier than the context
   budget implies. Understand this before the real run -- it may mean the
   long-context regime the experiment is *about* is not actually being
   exercised.

### Second 30s timeout, in rlm this time

```
rlm_train.repl.subprocess.WorkerStartupError:
Worker did not produce init line in 30.0s; stderr=''
```

Independent of the verifiers heartbeat: rlm's REPL subprocess gets 30s to emit
its init line, which is not enough when 8 rollouts spawn interpreters against a
box whose GPU is saturated. Cost 2 of 8 rollouts on step 0. Same shape as the
other one -- a fixed 30s budget that is fine on an idle machine and wrong under
load. Fix in the fork.

Also seen: `final_env_response returned raw dicts/strings instead of
vf.Messages`, the same wrong-type bug as `prompt_messages`.
