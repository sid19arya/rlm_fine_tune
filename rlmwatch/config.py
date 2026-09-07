"""Configuration: one YAML file, validated at load.

The rule this module enforces is from the spec: **no monitoring logic in the
training script**. Anything experiment-specific -- thresholds, metric key names,
budget, what to do on a terminal verdict -- is data here, not code there.

Validation is strict and happens at load time. A monitor that discovers its own
misconfiguration six hours in has already failed at its job.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

Regime = Literal["sft", "rl"]
TerminalAction = Literal["terminate", "stop"]

TERMINAL_ACTIONS = ("terminate", "stop")
REGIMES = ("sft", "rl")

# ${VAR} or ${VAR:-default}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """Raised for any malformed or internally inconsistent config."""


def _interpolate(value: str, env: dict[str, str], *, path: str) -> str:
    """Substitute ${VAR} from the environment.

    A missing variable with no default is an error rather than an empty string:
    silently monitoring pod "" is worse than refusing to start.
    """

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if env.get(name):
            return env[name]
        if default is not None:
            return default
        raise ConfigError(
            f"{path}: environment variable ${{{name}}} is unset and has no default. "
            f"Export it, or write ${{{name}:-<default>}}."
        )

    return _ENV_PATTERN.sub(replace, value)


def _walk_interpolate(node: Any, env: dict[str, str], path: str = "") -> Any:
    if isinstance(node, str):
        return _interpolate(node, env, path=path or "<root>")
    if isinstance(node, dict):
        return {
            k: _walk_interpolate(v, env, f"{path}.{k}" if path else str(k))
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_walk_interpolate(v, env, f"{path}[{i}]") for i, v in enumerate(node)]
    return node


@dataclass
class Expect:
    """What the pod is supposed to be. Checked by the startup gate."""

    gpu_type: str = ""
    gpu_count: int = 1
    min_free_vram_gb: float = 0.0
    min_disk_gb: float = 0.0
    checkpoint_dir: str = "/workspace/ckpt"

    def validate(self) -> None:
        if self.gpu_count < 1:
            raise ConfigError("expect.gpu_count must be >= 1")
        if self.min_free_vram_gb < 0 or self.min_disk_gb < 0:
            raise ConfigError("expect.min_free_vram_gb / min_disk_gb must be >= 0")


@dataclass
class Startup:
    deadline_s: float = 900.0
    on_failure: str = "terminate"

    def validate(self) -> None:
        if self.deadline_s <= 0:
            raise ConfigError("startup.deadline_s must be > 0")
        if self.on_failure not in TERMINAL_ACTIONS:
            raise ConfigError("startup.on_failure must be 'terminate' or 'stop'")


@dataclass
class StallThreshold:
    """Per-phase staleness budgets.

    Flat thresholds do not work for RL: a GRPO rollout can run for many minutes
    with nothing logged, so a threshold tight enough to catch a hung `update`
    false-alarms on every normal `rollout`.

    `auto` means "derive from the warm-up baseline measured by the startup
    gate": `auto_multiplier` x the measured phase duration, floored at
    `auto_floor_s`. Until a baseline exists, `auto` falls back to the floor
    rather than to `default` -- an unmeasured rollout phase should be given
    room, not tripped early.
    """

    default: float = 300.0
    auto_multiplier: float = 4.0
    auto_floor_s: float = 600.0
    phases: dict[str, Any] = field(default_factory=dict)

    def resolve(self, phase: str | None, baselines: dict[str, float]) -> float:
        if phase is None:
            return self.default
        setting = self.phases.get(phase)
        if setting is None:
            return self.default
        if setting == "auto":
            baseline = baselines.get(phase)
            if baseline is None:
                return self.auto_floor_s
            return max(self.auto_floor_s, self.auto_multiplier * baseline)
        return float(setting)

    def validate(self) -> None:
        if self.default <= 0:
            raise ConfigError("stall_threshold.default must be > 0")
        if self.auto_multiplier <= 0 or self.auto_floor_s <= 0:
            raise ConfigError("stall_threshold.auto_multiplier / auto_floor_s must be > 0")
        for name, value in self.phases.items():
            if value == "auto":
                continue
            if not isinstance(value, (int, float)) or value <= 0:
                raise ConfigError(
                    f"stall_threshold.{name} must be a positive number or 'auto', "
                    f"got {value!r}"
                )


@dataclass
class Health:
    """Metric-shape thresholds (sections 4.2 and 5).

    Defaults match the spec's template. `reward_std_min` and its patience are
    the highest-value pair in this file: zero within-group reward variance means
    the advantage is zero and training is a no-op, while every dashboard
    continues to look completely normal.
    """

    reward_std_min: float = 0.01
    reward_std_patience: int = 20
    kl_max: float = 0.15
    entropy_min: float = 0.3
    format_success_min: float = 0.9
    clip_max: float = 0.3
    completion_len: tuple = (64.0, 4096.0)
    throughput_degradation_pct: float = 30.0
    grad_norm_min: float = 1e-6
    grad_norm_max: float = 1e3
    plateau_window: int = 50
    eval_patience: int = 3
    rollout_duration_multiplier: float = 4.0
    seq_len_p95_warn_pct: float = 90.0
    gpu_util_min_pct: float = 5.0
    gpu_util_window_s: float = 600.0
    # Canonical probe metric name -> the key this trainer actually emits.
    # TRL / verl / OpenRLHF all use different names for the same quantity;
    # map them here rather than patching the trainer.
    metric_aliases: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not 0 <= self.format_success_min <= 1:
            raise ConfigError("health.format_success_min must be in [0, 1]")
        if len(self.completion_len) != 2:
            raise ConfigError("health.completion_len must be a [min, max] pair")
        lo, hi = self.completion_len
        if lo >= hi:
            raise ConfigError(
                f"health.completion_len must be [min, max] with min < max, got [{lo}, {hi}]"
            )
        if self.reward_std_patience < 1:
            raise ConfigError("health.reward_std_patience must be >= 1")
        if self.grad_norm_min >= self.grad_norm_max:
            raise ConfigError("health.grad_norm_min must be < health.grad_norm_max")

    def key_for(self, canonical: str) -> str:
        """Translate a canonical metric name into this trainer's key."""
        return self.metric_aliases.get(canonical, canonical)


@dataclass
class Budget:
    hourly_rate_usd: float = 0.0
    max_usd: float = 0.0
    warn_pct: float = 75.0
    max_wall_clock_h: float = 24.0
    storage_gb: float = 0.0
    storage_usd_per_gb_month: float = 0.07

    def validate(self) -> None:
        if self.hourly_rate_usd < 0:
            raise ConfigError("budget.hourly_rate_usd must be >= 0")
        if self.max_usd <= 0:
            raise ConfigError(
                "budget.max_usd must be > 0 -- the cap is unconditional and there is "
                "deliberately no 'unlimited' setting"
            )
        if not 0 < self.warn_pct < 100:
            raise ConfigError("budget.warn_pct must be in (0, 100)")
        if self.max_wall_clock_h <= 0:
            raise ConfigError("budget.max_wall_clock_h must be > 0")

    def spend_at(self, elapsed_h: float) -> float:
        """Accrued spend: compute + storage.

        Billing starts when the pod is *provisioned*, not when training starts,
        so callers must pass elapsed time since pod creation.
        """
        storage = self.storage_gb * self.storage_usd_per_gb_month * (elapsed_h / 730.0)
        return self.hourly_rate_usd * elapsed_h + storage


@dataclass
class Failsafe:
    confirm_delay_s: float = 120.0
    unknown_tolerance_s: float = 900.0
    dead_mans_timeout_s: float = 1800.0
    checkpoint_timeout_s: float = 600.0
    on_terminal: str = "terminate"
    # Terminate destroys every disk that is not a network volume. Requiring an
    # explicit acknowledgement here is what stops `on_terminal: terminate` from
    # being an accidental default that eats an un-uploaded adapter.
    checkpoint_dir_is_network_volume: bool = False

    def validate(self) -> None:
        if self.on_terminal not in TERMINAL_ACTIONS:
            raise ConfigError("failsafe.on_terminal must be 'terminate' or 'stop'")
        for name in (
            "confirm_delay_s",
            "unknown_tolerance_s",
            "dead_mans_timeout_s",
            "checkpoint_timeout_s",
        ):
            if getattr(self, name) <= 0:
                raise ConfigError(f"failsafe.{name} must be > 0")
        if self.dead_mans_timeout_s <= self.confirm_delay_s:
            raise ConfigError(
                "failsafe.dead_mans_timeout_s must exceed confirm_delay_s, otherwise the "
                "dead-man's switch fires before a fail verdict can be confirmed"
            )


@dataclass
class Notify:
    slack_webhook: str = ""
    heartbeat_url: str = ""
    webhook: str = ""
    console: bool = True
    # Nous Hermes Agent inbound webhook: http://<host>:8644/webhooks/<route>.
    # The secret comes from the environment, never the file -- a signing key in
    # a committed config is a key that is no longer secret.
    hermes_webhook: str = ""
    hermes_secret: str = ""

    def validate(self) -> None:
        if self.hermes_webhook and not self.hermes_secret:
            raise ConfigError(
                "notify.hermes_webhook is set but notify.hermes_secret is empty. "
                "The Hermes gateway rejects unsigned webhooks, so this would fail "
                "silently at the first alert. Export HERMES_WEBHOOK_SECRET."
            )
        for name in ("slack_webhook", "heartbeat_url", "webhook", "hermes_webhook"):
            url = getattr(self, name)
            if url and not url.startswith(("http://", "https://")):
                raise ConfigError(f"notify.{name} must be an http(s) URL, got {url!r}")


@dataclass
class Run:
    name: str = ""
    regime: str = "rl"
    wandb: str = ""
    pod_id: str = ""

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("run.name is required")
        if self.regime not in REGIMES:
            raise ConfigError(f"run.regime must be 'sft' or 'rl', got {self.regime!r}")
        if self.wandb and self.wandb.count("/") != 2:
            raise ConfigError(f"run.wandb must be 'entity/project/run-id', got {self.wandb!r}")

    @property
    def wandb_url(self) -> str:
        if not self.wandb:
            return ""
        entity, project, run_id = self.wandb.split("/")
        return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


@dataclass
class RunConfig:
    run: Run = field(default_factory=Run)
    expect: Expect = field(default_factory=Expect)
    startup: Startup = field(default_factory=Startup)
    stall_threshold: StallThreshold = field(default_factory=StallThreshold)
    health: Health = field(default_factory=Health)
    budget: Budget = field(default_factory=Budget)
    failsafe: Failsafe = field(default_factory=Failsafe)
    notify: Notify = field(default_factory=Notify)
    # Sentinel poll interval. Kept well above 1/s: the entire external observer
    # has to cost under $1/day including API calls.
    poll_interval_s: float = 60.0

    def validate(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if is_dataclass(value) and hasattr(value, "validate"):
                value.validate()
        if self.poll_interval_s <= 0:
            raise ConfigError("poll_interval_s must be > 0")
        if (
            self.failsafe.on_terminal == "terminate"
            and not self.failsafe.checkpoint_dir_is_network_volume
        ):
            raise ConfigError(
                "failsafe.on_terminal is 'terminate' but "
                "failsafe.checkpoint_dir_is_network_volume is false. Terminate deletes "
                "every disk that is not a network volume, so this combination throws "
                "away the checkpoints. Either mount expect.checkpoint_dir on a network "
                "volume and set the flag, or use on_terminal: stop and accept that "
                "storage keeps billing."
            )


# --- loading -----------------------------------------------------------------

_SECTIONS: dict[str, Any] = {
    "run": Run,
    "expect": Expect,
    "startup": Startup,
    "health": Health,
    "budget": Budget,
    "failsafe": Failsafe,
    "notify": Notify,
}


def _coerce(default: Any, value: Any, path: str) -> Any:
    """Coerce a YAML scalar to the shape implied by the field's default."""
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if isinstance(default, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        return tuple(float(v) for v in value)
    if isinstance(default, dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        return value
    if isinstance(default, int) and not isinstance(default, bool):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if isinstance(default, str):
        if isinstance(value, (dict, list)):
            raise ConfigError(f"{path}: expected a string, got {type(value).__name__}")
        return str(value)
    return value


def _build(cls: Any, data: dict[str, Any], path: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {unknown}. Known keys: {sorted(known)}. "
            "A typo in a monitoring config is a silent failure, so it is rejected here."
        )
    template = cls()
    kwargs = {
        name: _coerce(getattr(template, name), data[name], f"{path}.{name}")
        for name in known
        if name in data
    }
    return cls(**kwargs)


def from_dict(data: dict[str, Any]) -> RunConfig:
    """Build and validate a RunConfig from an already-interpolated mapping."""
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")

    unknown = sorted(set(data) - set(_SECTIONS) - {"stall_threshold", "poll_interval_s"})
    if unknown:
        raise ConfigError(f"unknown top-level section(s): {unknown}")

    kwargs: dict[str, Any] = {}
    for key, cls in _SECTIONS.items():
        if key in data:
            section = data[key]
            if not isinstance(section, dict):
                raise ConfigError(f"{key}: expected a mapping, got {type(section).__name__}")
            kwargs[key] = _build(cls, section, key)

    if "stall_threshold" in data:
        raw = data["stall_threshold"]
        if not isinstance(raw, dict):
            raise ConfigError("stall_threshold: expected a mapping")
        st = StallThreshold()
        for key, value in raw.items():
            if key in ("default", "auto_multiplier", "auto_floor_s"):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ConfigError(f"stall_threshold.{key}: expected a number, got {value!r}")
                setattr(st, key, float(value))
            else:
                st.phases[key] = value if value == "auto" else value
        kwargs["stall_threshold"] = st

    if "poll_interval_s" in data:
        kwargs["poll_interval_s"] = float(data["poll_interval_s"])

    cfg = RunConfig(**kwargs)
    cfg.validate()
    return cfg


def load_config(path: str | Path, env: dict[str, str] | None = None) -> RunConfig:
    """Load YAML, interpolate ${ENV} references, validate, return."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if raw is None:
        raise ConfigError(f"{path}: file is empty")
    resolved = _walk_interpolate(raw, dict(os.environ) if env is None else dict(env))
    return from_dict(resolved)


__all__ = [
    "Budget",
    "ConfigError",
    "Expect",
    "Failsafe",
    "Health",
    "Notify",
    "Run",
    "RunConfig",
    "StallThreshold",
    "Startup",
    "from_dict",
    "load_config",
]
