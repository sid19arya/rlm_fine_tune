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
