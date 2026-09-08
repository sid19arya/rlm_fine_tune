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
