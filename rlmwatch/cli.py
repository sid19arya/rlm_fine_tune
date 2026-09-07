"""`rlmwatch` command line.

    rlmwatch preflight -c cfg.yaml    # run the startup gate, report, exit
    rlmwatch watch     -c cfg.yaml    # run the sentinel (long-lived or --once)
    rlmwatch snapshot  -c cfg.yaml    # collect a diagnostic bundle now
    rlmwatch kill      -c cfg.yaml    # stop or terminate the pod, explicitly

`watch --once` is the cron form: one sweep per invocation, no long-lived process
to supervise. `--dry-run` caps the ladder at L1 so it alerts but never kills --
the way to earn confidence in a config before it holds the kill switch on a live
run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

from rlmwatch.config import ConfigError, load_config

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CONFIG = 2


def _log_setup(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(
            f"{name} is not set. rlmwatch reads credentials from the environment only, "
            f"never from the config file, so that a config can be committed."
        )
    return value


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))


def cmd_preflight(args: argparse.Namespace) -> int:
    """Run the startup gate. Intended to be the first thing the pod does."""
    from rlmwatch.actions import EscalationLadder
    from rlmwatch.clients.runpod import RunPodClient
    from rlmwatch.clients.wandb import WandbClient
    from rlmwatch.diagnostics import Diagnostics
    from rlmwatch.notify import Notifier
    from rlmwatch.probes.base import Context
    from rlmwatch.probes.startup import (
        StartupContext,
        StartupHooks,
        SystemHardware,
        gate,
        gate_and_enforce,
    )

    cfg = load_config(args.config)
    runpod = RunPodClient(_require_env("RUNPOD_API_KEY"))
    wandb_client = WandbClient(os.environ.get("WANDB_API_KEY"))
    notifier = Notifier(cfg)

    sctx = StartupContext(
        ctx=Context(cfg=cfg, wandb=wandb_client, runpod=runpod),
        hardware=SystemHardware(),
        # The CLI cannot supply model/data/warm-up hooks -- those live in the
        # training script. Their checks report `unknown` and are listed as
        # unwired rather than silently counted as passing.
        hooks=StartupHooks(),
        notifier=notifier,
    )

    if args.no_enforce:
        result = gate(sctx)
    else:
        ladder = EscalationLadder(
            cfg, notifier=notifier,
            diagnostics=Diagnostics(cfg, runpod=runpod, wandb=wandb_client),
            runpod=runpod,
        )
        result = gate_and_enforce(sctx, ladder=ladder)

    for verdict in result.verdicts:
        print(verdict)
    print(result.summary())
    if result.unwired:
        print(f"not wired (reported, not passed): {', '.join(result.unwired)}")
    _emit(result.as_dict(), args.json)
    return EXIT_OK if result.passed else EXIT_FAIL


def cmd_watch(args: argparse.Namespace) -> int:
    """Run the sentinel. Must not be run on the pod it is watching."""
    from rlmwatch.sentinel import build_sentinel

    cfg = load_config(args.config)
    sentinel = build_sentinel(
        cfg,
        runpod_api_key=_require_env("RUNPOD_API_KEY"),
        wandb_api_key=os.environ.get("WANDB_API_KEY"),
        pod_created_at=args.pod_created_at,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("DRY RUN: the ladder will alert but will not stop or terminate anything.")

    results = sentinel.run(max_ticks=1 if args.once else args.max_ticks)
    for result in results:
        for verdict in result.verdicts:
            print(verdict)
    if results:
        _emit(results[-1].as_dict(), args.json)
        return EXIT_FAIL if results[-1].worst_status == "fail" else EXIT_OK
    return EXIT_OK


def cmd_digest(args: argparse.Namespace) -> int:
    """Build a training-dynamics digest and optionally send it to Hermes.

    Separate from `watch` on purpose. `watch` is exception-based and should
    stay quiet on a healthy run; a digest is periodic and speaks whether or not
    anything is wrong. Running them on separate cadences means a 30-minute
    digest never delays a stall alert, and a healthy run never generates noise
    just because the digest is due.
    """
    from rlmwatch.clients.runpod import RunPodClient
    from rlmwatch.clients.wandb import WandbClient
    from rlmwatch.config import from_dict
    from rlmwatch.digest import build_digest
    from rlmwatch.notify import HermesSink
    from rlmwatch.probes.base import Context

    if args.config:
        cfg = load_config(args.config)
    elif args.run:
        # Standalone mode, for an observer that pip-installed rlmwatch and has
        # no checkout: `configs/` is not part of the wheel. Everything the
        # digest needs is the run path and the rate, so it is accepted directly
        # rather than forcing a config file to be copied around and drift.
        cfg = from_dict({
            "run": {"name": args.run.split("/")[-1], "wandb": args.run},
            "budget": {"hourly_rate_usd": args.rate, "max_usd": args.max_usd},
            "failsafe": {"on_terminal": "stop"},
            "notify": {"console": False},
        })
    else:
        print("digest needs either -c/--config or --run entity/project/run-id",
              file=sys.stderr)
        return EXIT_CONFIG
    runpod = None
    if cfg.run.pod_id and os.environ.get("RUNPOD_API_KEY"):
        runpod = RunPodClient(os.environ["RUNPOD_API_KEY"])
    ctx = Context(
        cfg=cfg,
        wandb=WandbClient(os.environ.get("WANDB_API_KEY")),
        runpod=runpod,
        local={"rollouts_per_step": args.rollouts_per_step} if args.rollouts_per_step
        else {},
    )

    digest = build_digest(ctx, rollouts_per_step=args.rollouts_per_step,
                          billing_lead_s=args.billing_lead_min * 60.0)
    print(digest.as_text())
    _emit(digest.as_dict(), args.json)

    if args.send:
        if not cfg.notify.hermes_webhook:
            print("--send needs notify.hermes_webhook in the config", file=sys.stderr)
            return EXIT_CONFIG
        sink = HermesSink(cfg.notify.hermes_webhook, cfg.notify.hermes_secret)
        payload = {
            "event_type": "rlmwatch.digest",
            "source": "rlmwatch",
            "run": cfg.run.name,
            "summary": digest.headline(),
            "text": digest.as_text(),
            **digest.as_dict(),
        }
        if sink.post(payload):
            print("\nsent to Hermes", file=sys.stderr)
        else:
            print("\nHermes rejected the digest -- check the route and the secret",
                  file=sys.stderr)
            return EXIT_FAIL

    # A digest never fails a run. It reports; the ladder decides.
    return EXIT_OK


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Collect a diagnostic bundle on demand, without touching the run."""
    from rlmwatch.clients.runpod import RunPodClient
    from rlmwatch.clients.wandb import WandbClient
    from rlmwatch.diagnostics import Diagnostics

    cfg = load_config(args.config)
    diagnostics = Diagnostics(
        cfg,
        runpod=RunPodClient(_require_env("RUNPOD_API_KEY")),
        wandb=WandbClient(os.environ.get("WANDB_API_KEY")),
    )
    snapshot = diagnostics.snapshot(args.reason)
    path = snapshot.write(args.out)
    print(f"{snapshot.summary()} -> {path}")
    if snapshot.errors:
        for name, error in snapshot.errors.items():
            print(f"  collector {name} failed: {error}", file=sys.stderr)
    return EXIT_OK


def cmd_kill(args: argparse.Namespace) -> int:
    """Stop or terminate the pod, explicitly.

    `stop` and `terminate` are different operations and this command makes you
    say which: stop pauses compute and keeps billing storage, terminate deletes
    every disk that is not a network volume.
    """
    from rlmwatch.clients.runpod import RunPodClient

    cfg = load_config(args.config)
    pod_id = args.pod_id or cfg.run.pod_id
    if not pod_id:
        print("no pod_id in config and none given with --pod-id", file=sys.stderr)
        return EXIT_CONFIG

    action = args.action or cfg.failsafe.on_terminal
    runpod = RunPodClient(_require_env("RUNPOD_API_KEY"))

    if action == "terminate" and not args.yes:
        print(
            f"terminate deletes pod {pod_id} and every disk on it that is not a network "
            f"volume. Re-run with --yes, or use --action stop to pause compute and keep "
            f"the disk (storage keeps billing).",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    if action == "terminate":
        runpod.terminate(pod_id)
    else:
        runpod.stop(pod_id)
    print(f"{action} sent for pod {pod_id}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlmwatch",
        description="Monitoring and failsafes for RLM fine-tuning runs on RunPod.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="also emit machine-readable JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config(sub):
        sub.add_argument("-c", "--config", required=True, help="path to the run YAML")
        return sub

    preflight = add_config(subparsers.add_parser(
        "preflight", help="run the startup gate before training begins"))
    preflight.add_argument(
        "--no-enforce", action="store_true",
        help="report only; do not stop or terminate the pod on failure",
    )
    preflight.set_defaults(func=cmd_preflight)

    watch = add_config(subparsers.add_parser(
        "watch", help="run the external sentinel (never on the pod it watches)"))
    watch.add_argument("--once", action="store_true",
                       help="single sweep and exit -- the cron form")
    watch.add_argument("--max-ticks", type=int, default=None)
    watch.add_argument("--pod-created-at", type=float, default=None,
                       help="unix time the pod was provisioned; billing starts there, "
                            "not when training does")
    watch.add_argument("--dry-run", action="store_true",
                       help="cap the ladder at L1: alert, never kill")
    watch.set_defaults(func=cmd_watch)

    # Not add_config(): the digest is the one command an external observer
    # runs with no checkout, so --run replaces the config file entirely.
    digest = subparsers.add_parser(
        "digest", help="report training dynamics (what moved), optionally to Hermes")
    digest.add_argument("-c", "--config", default=None,
                        help="run YAML; omit it and pass --run instead")
    digest.add_argument("--run", default=None,
                        help="W&B run path entity/project/run-id, for use without a "
                             "config file")
    digest.add_argument("--rate", type=float, default=0.0,
                        help="pod cost per hour, used for the spend line")
    digest.add_argument("--max-usd", type=float, default=5.0,
                        help="budget cap, reported alongside spend")
    digest.add_argument("--send", action="store_true",
                        help="POST the digest to the configured Hermes webhook")
    digest.add_argument("--rollouts-per-step", type=int, default=None,
                        help="used for the reward noise band; V0 is 32")
    digest.add_argument("--billing-lead-min", type=float, default=0.0,
                        help="minutes the pod was billing before the run started "
                             "(provisioning + setup). Only used when RunPod cannot "
                             "be read, to keep the spend figure honest.")
    digest.set_defaults(func=cmd_digest)

    snapshot = add_config(subparsers.add_parser(
        "snapshot", help="collect a diagnostic bundle now"))
    snapshot.add_argument("--reason", default="manual")
    snapshot.add_argument("--out", default="snapshots")
    snapshot.set_defaults(func=cmd_snapshot)

    kill = add_config(subparsers.add_parser("kill", help="stop or terminate the pod"))
    kill.add_argument("--pod-id", default=None)
    kill.add_argument("--action", choices=("stop", "terminate"), default=None)
    kill.add_argument("--yes", action="store_true",
                      help="required to confirm terminate, which deletes the disks")
    kill.set_defaults(func=cmd_kill)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _log_setup(args.verbose)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
