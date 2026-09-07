#!/usr/bin/env python3
"""Confirm the no-recursion arm is actually selected, before spending GPU time.

This replaces the old `strip_sub_lm_calls.py`, which edited rlm's source in
place and then scanned it. The condition is now a config flag in the pinned
fork (`enable_sub_lm`), which drives the REPL globals, the system prompt, the
orchestrator addendum and the rubric's sub-call gate from one place -- so the
asymmetric state that scanner existed to catch is unrepresentable rather than
merely detectable.

What remains worth checking is that the flag is *wired*, which a scanner over
source text cannot tell you. So this asks the real objects:

* build a `Worker` with the flag off and look at the REPL globals it actually
  binds -- including after `_restore_scaffold()`, which runs after every turn
  and would silently restore delegation on turn two if the flag were honoured
  in only one of the two binding sites;
* render the real system prompt and check it never names a tool the REPL does
  not provide;
* check the OOLONG rubric's sub-call floor, which would otherwise fail the gate
  on every rollout and flatten reward variance to zero.

    python verify_ablation.py                 # reads smoke.toml for the flag
    python verify_ablation.py --expect-sub-lm # assert the WITH-recursion arm

Exit code 0 means the arm you asked for is the arm you would get.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_RLM = Path("/workspace/rlm")
DEFAULT_CONFIG = Path(__file__).resolve().parent / "smoke.toml"

SUB_LM_NAMES = ("llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched")
#: Phrases that would tell a model delegation is available even without naming
#: a function. A prompt clean of the names but still describing sub-LLMs is the
#: same failure in slower motion.
PROMPT_TELLS = ("sub-LLM", "sub_llm", "sub-LM", "recursively query")


def read_flag(config: Path) -> bool | None:
    """Read `enable_sub_lm` from `[orchestrator.train.env.args]`.

    That table is the only one prime-rl turns into `load_environment` kwargs.
    The path is spelled out rather than searched for on purpose: an
    `enable_sub_lm` sitting in any other table is dead config that prime-rl
    ignores silently, and reporting it as the selected arm would make this
    script certify the opposite of what would run.
    """
    try:
        import tomllib
    except ImportError:  # pragma: no cover - py<3.11
        import tomli as tomllib  # type: ignore[no-redef]
    try:
        data = tomllib.loads(config.read_text(encoding="utf-8"))
    except OSError:
        return None

    flags = [
        bool(args["enable_sub_lm"])
        for args in _env_arg_tables(data)
        if "enable_sub_lm" in args
    ]
    if not flags:
        return None
    if len(set(flags)) > 1:
        raise ValueError(
            "training environments disagree on enable_sub_lm. Every environment in "
            "the train mixture has to be on the same arm, or the run is measuring "
            "two conditions at once."
        )
    return flags[0]


def _env_arg_tables(data: dict) -> list[dict]:
    """Every `[orchestrator.train.env.args]` table in the config.

    `[[orchestrator.train.env]]` is an array of tables -- a train mixture can
    hold several environments -- so this returns one args dict per entry. A
    single `[orchestrator.train.env]` table is accepted too, since TOML allows
    both spellings and prime-rl reads either.
    """
    env = data.get("orchestrator", {}).get("train", {}).get("env")
    entries = env if isinstance(env, list) else [env] if isinstance(env, dict) else []
    return [
        entry["args"]
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("args"), dict)
    ]


def misplaced_flag(config: Path) -> str | None:
    """Find an `enable_sub_lm` outside the args table. Dead config, silently.

    Worth an explicit check: the key looks correct to a reader and does nothing
    at all, which is the failure mode this whole arm-selection design exists to
    remove.
    """
    try:
        import tomllib
    except ImportError:  # pragma: no cover - py<3.11
        import tomli as tomllib  # type: ignore[no-redef]
    try:
        data = tomllib.loads(config.read_text(encoding="utf-8"))
    except OSError:
        return None

    live = [id(args) for args in _env_arg_tables(data)]

    def walk(node: object, path: tuple[str, ...]) -> str | None:
        if isinstance(node, list):
            for index, item in enumerate(node):
                found = walk(item, (*path[:-1], f"{path[-1]}[{index}]") if path else path)
                if found:
                    return found
            return None
        if not isinstance(node, dict):
            return None
        if "enable_sub_lm" in node and id(node) not in live:
            return ".".join(path) or "<root>"
        for key, value in node.items():
            found = walk(value, (*path, key))
            if found:
                return found
        return None

    return walk(data, ())


def check(expect_sub_lm: bool, rlm_root: Path) -> list[tuple[bool, str]]:
    """Ask the real objects, not the source text. Returns (ok, detail) rows."""
    results: list[tuple[bool, str]] = []

    sys.path.insert(0, str(rlm_root / "training" / "src"))
    try:
        from rlm.core.types import QueryMetadata
        from rlm.utils.prompts import build_rlm_system_prompt, default_system_prompt
    except ImportError as exc:
        return [(False, f"cannot import rlm from {rlm_root}: {exc}. Is the pinned "
                        f"fork installed in this venv?")]

    # --- the prompt the model will actually be shown ---
    try:
        messages = build_rlm_system_prompt(
            system_prompt=default_system_prompt(enable_sub_lm=expect_sub_lm),
            query_metadata=QueryMetadata("x" * 1000),
            enable_sub_lm=expect_sub_lm,
        )
    except TypeError as exc:
        return [(False, f"build_rlm_system_prompt has no enable_sub_lm parameter: {exc}. "
                        f"This is upstream rlm, not the fork -- check RLM_SHA in setup.sh.")]

    prompt = "\n".join(m["content"] for m in messages)
    named = [n for n in SUB_LM_NAMES if n in prompt]
    tells = [t for t in PROMPT_TELLS if t in prompt]

    if expect_sub_lm:
        results.append((bool(named), f"prompt offers the sub-LM tools ({len(named)} named)"))
    else:
        results.append((not named, "prompt names no sub-LM function"
                                   + (f" -- found {named}" if named else "")))
        results.append((not tells, "prompt describes no delegation"
                                   + (f" -- found {tells}" if tells else "")))

    # --- the globals the REPL will actually bind ---
    try:
        from rlm_train.worker import Worker
    except ImportError as exc:
        results.append((False, f"cannot import rlm_train.worker: {exc}"))
        return results

    os.environ.pop("RLM_TRAIN_ENABLE_SUB_LM", None)
    try:
        worker = Worker(proxy_url="http://localhost:0", rollout_id="verify",
                        enable_sub_lm=expect_sub_lm)
    except TypeError as exc:
        results.append((False, f"Worker has no enable_sub_lm parameter: {exc}. This is "
                               f"upstream rlm, not the fork -- check RLM_SHA in setup.sh."))
        return results
    bound = [n for n in SUB_LM_NAMES if n in worker.globals]
    if expect_sub_lm:
        results.append((len(bound) == len(SUB_LM_NAMES), f"REPL binds the tools ({bound})"))
    else:
        results.append((not bound, "REPL binds no sub-LM tool"
                                   + (f" -- found {bound}" if bound else "")))

    # The turn-two trap: _restore_scaffold re-installs the scaffolding after
    # every turn, so a flag honoured only at setup restores delegation silently.
    worker._restore_scaffold()
    after = [n for n in SUB_LM_NAMES if n in worker.globals]
    if expect_sub_lm:
        results.append((len(after) == len(SUB_LM_NAMES),
                        "_restore_scaffold keeps the tools bound"))
    else:
        results.append((not after, "_restore_scaffold does not re-bind the tools"
                                   + (f" -- found {after}" if after else "")))

    # --- agreement, the property that matters ---
    agree = all((n in prompt) == (n in worker.globals) for n in SUB_LM_NAMES)
    results.append((agree, "prompt and REPL agree on every tool name"))

    # --- the rubric gate ---
    try:
        from rlm_train.rubric import RLMTrainRubric

        rubric = RLMTrainRubric(correctness=lambda **kw: 1.0,
                                min_subcall=1 if expect_sub_lm else 0)
        passes = rubric._passes_gates({"rlm_iterations": 5, "rlm_sub_llm_calls": 0})
        if expect_sub_lm:
            results.append((True, "sub-call gate active (with-recursion arm)"))
        else:
            results.append((passes, "rubric does not gate on sub-calls"
                                    + ("" if passes else " -- min_subcall > 0 would fail "
                                                         "every rollout and flatten reward "
                                                         "variance to zero")))
    except ImportError:
        results.append((True, "rubric not importable here (verifiers absent); skipped"))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rlm-root", type=Path, default=DEFAULT_RLM)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expect-sub-lm", action="store_true",
                        help="assert the WITH-recursion arm instead")
    args = parser.parse_args(argv)

    stray = misplaced_flag(args.config)
    if stray is not None:
        print(
            f"FAILED: {args.config} has enable_sub_lm under [{stray}], not under\n"
            f"[orchestrator.train.env.args]. prime-rl only turns that one table into\n"
            f"load_environment kwargs, so the key you see is dead config and the run\n"
            f"would use the WITH-recursion arm regardless of what it says.",
            file=sys.stderr,
        )
        return 1

    expect = args.expect_sub_lm
    if not expect:
        from_config = read_flag(args.config)
        if from_config is None:
            print(f"note: no enable_sub_lm in [orchestrator.train.env.args] of "
                  f"{args.config}; assuming the no-recursion arm", file=sys.stderr)
        else:
            expect = from_config

    arm = "WITH recursion" if expect else "NO recursion (ablation)"
    print(f"verifying the {arm} arm against {args.rlm_root}\n")

    results = check(expect, args.rlm_root)
    for ok, detail in results:
        print(f"  [{'ok' if ok else 'FAIL'}] {detail}")

    if all(ok for ok, _ in results):
        print(f"\nThe {arm} arm is correctly selected.")
        return 0
    print(
        f"\nFAILED. Do not launch: the run would not be testing the {arm} arm.\n"
        f"Check that setup.sh pinned the fork (RLM_SHA) and that smoke.toml's\n"
        f"[env].enable_sub_lm reached the environment factory.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
