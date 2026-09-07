#!/usr/bin/env python3
"""Remove sub-LM calls from the REPL globals **and** the system prompt.

This is the experiment. The question V0 is plumbing for is whether RL improves a
model's use of a Python REPL over long context *with recursion removed* -- so a
surviving sub-LM call does not merely add noise, it answers a different
question.

The failure this script exists to prevent is asymmetric removal. Strip the REPL
global but leave the prompt mention and the model keeps emitting `llm_query(...)`
calls, every rollout dies on `NameError`, the scorer returns zeros, and from
outside the run looks completely healthy: GPUs busy, steps advancing, loss
curve plausible. That is strictly worse than not stripping at all, because it
produces a clean-looking null result rather than an obvious crash.

    python strip_sub_lm_calls.py --verify           # report, change nothing
    python strip_sub_lm_calls.py --apply            # edit the globals, then verify
    python strip_sub_lm_calls.py --verify --json    # machine-readable

`--apply` edits **code bindings only** -- imports, `def`s, assignments, and
entries in a REPL globals dict. It deliberately refuses to touch prompt text,
because commenting out a line inside a triple-quoted string does not remove it
from the prompt: it inserts a `#` into the string and the model still reads
`llm_query`. Prompt wording has to be rewritten by a human who can say what the
sentence should be instead.

Editing is otherwise conservative: lines are commented out with a marker rather
than deleted, and a `.bak` is written.

**Read the diff after `--apply`.** Commenting out a whole line is right for an
import or a single dict entry, but if the globals dict is written on one line
the edit comments out the entire literal and the function silently returns
`None`. The `.bak` is there for exactly that; a quick `git diff` before
launching costs seconds and catches it.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ROOT = Path("/workspace/rlm/training/src/rlm_train")

MARKER = "# rlmwatch: sub-LM call removed for the V0 no-recursion condition"

#: Names that constitute a sub-LM call. Matching is intentionally broad -- a
#: false positive costs one manual review, a false negative costs the run.
SUB_LM_NAMES = (
    "llm_query",
    "llm_call",
    "sub_llm",
    "sub_lm",
    "recursive_llm",
    "call_llm",
    "ask_llm",
    "query_model",
)

#: Prompt text that tells the model a sub-LM is available. These are the ones
#: that produce NameError rollouts when the global is gone but the prompt is not.
PROMPT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bllm_query\b",
        r"\bsub-?LM\b",
        r"\bsub-?language model\b",
        r"you (?:can|may) (?:also )?(?:call|query|ask) (?:a|another) (?:language )?model",
        r"recursively (?:call|query|invoke)",
    )
)

CODE_SUFFIXES = (".py",)
PROMPT_SUFFIXES = (".py", ".txt", ".md", ".jinja", ".j2", ".yaml", ".yml", ".toml")


@dataclass
class Hit:
    path: Path
    line_no: int
    line: str
    kind: str  # "global" | "prompt"

    def as_dict(self) -> dict:
        return {"path": str(self.path), "line": self.line_no, "kind": self.kind,
                "text": self.line.strip()[:160]}


@dataclass
class Report:
    globals_: list[Hit] = field(default_factory=list)
    prompts: list[Hit] = field(default_factory=list)
    scanned: int = 0

    @property
    def clean(self) -> bool:
        return not self.globals_ and not self.prompts

    @property
    def asymmetric(self) -> bool:
        """One side stripped and not the other -- the dangerous state."""
        return bool(self.globals_) != bool(self.prompts)

    def as_dict(self) -> dict:
        return {
            "clean": self.clean,
            "asymmetric": self.asymmetric,
            "files_scanned": self.scanned,
            "globals": [h.as_dict() for h in self.globals_],
            "prompts": [h.as_dict() for h in self.prompts],
        }


def _is_commented(line: str) -> bool:
    return line.lstrip().startswith("#")


def prompt_block_lines(source: str) -> set[int]:
    """Line numbers inside a **multi-line** Python string -- i.e. prompt text.

    Needed because a `#` at the start of such a line is not a comment, it is
    text the model reads. Without this, commenting out a prompt line would make
    the scanner report "clean" while `llm_query` was still sitting in the system
    prompt -- precisely the asymmetric failure this script exists to prevent.

    Multi-line only, deliberately. A single-line string is usually a dict key or
    a short literal: treating `{"llm_query": fn}` as prompt text would hide the
    most important globals binding there is.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            start = node.lineno
            end = getattr(node, "end_lineno", start) or start
            if end > start:
                lines.update(range(start, end + 1))
    return lines


def _is_definition_or_binding(line: str, name: str) -> bool:
    """Is this line *providing* the sub-LM, rather than merely mentioning it?

    Binding into a REPL globals dict, defining the function, importing it, or
    assigning it. These are the lines that make the call available.
    """
    stripped = line.strip()
    patterns = (
        rf"^def\s+{name}\b",
        rf"^async\s+def\s+{name}\b",
        rf"^{name}\s*=",
        rf"^from\s+\S+\s+import\s+.*\b{name}\b",
        rf"^import\s+.*\b{name}\b",
        rf"[\"']{name}[\"']\s*:",             # {"llm_query": fn} dict literal
        rf"\[\s*[\"']{name}[\"']\s*\]\s*=",   # env["llm_query"] = fn
        rf"\bsetattr\([^,]+,\s*[\"']{name}[\"']",
        rf"\bupdate\(\s*\{{[^}}]*[\"']{name}[\"']",
    )
    return any(re.search(p, stripped) for p in patterns)


def scan(root: Path) -> Report:
    report = Report()
    if not root.exists():
        raise FileNotFoundError(
            f"{root} does not exist. Pass --root explicitly if the rlm checkout is "
            f"somewhere else; scanning the wrong tree and reporting 'clean' is the "
            f"single worst outcome this script can produce."
        )

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in PROMPT_SUFFIXES:
            continue
        if MARKER in path.name or ".bak" in path.suffixes:
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = source.splitlines()
        report.scanned += 1

        # In a .py file, only lines outside string literals can be "commented
        # out". Inside a docstring or a prompt constant, a leading `#` is text
        # the model reads. Non-.py files are prompt text end to end.
        in_string = (prompt_block_lines(source) if path.suffix in CODE_SUFFIXES
                     else set(range(1, len(lines) + 2)))

        for index, line in enumerate(lines, start=1):
            is_prompt_text = index in in_string
            if not is_prompt_text and (_is_commented(line) or MARKER in line):
                continue

            if (path.suffix in CODE_SUFFIXES and not is_prompt_text and any(
                _is_definition_or_binding(line, name) for name in SUB_LM_NAMES
            )):
                report.globals_.append(Hit(path, index, line, "global"))
                continue

            if any(pattern.search(line) for pattern in PROMPT_PATTERNS):
                report.prompts.append(Hit(path, index, line, "prompt"))

    return report


def apply(report: Report) -> int:
    """Comment out the code bindings, leaving a .bak and an explanatory marker.

    Prompt hits are never edited. Commenting out a line inside a triple-quoted
    string does not remove it from the prompt -- it puts a `#` in the string and
    the model still reads `llm_query`. Worse, the line then *looks* handled to a
    naive scanner. Prompt wording needs a human who can decide what the sentence
    should say instead.
    """
    by_file: dict[Path, list[Hit]] = {}
    for hit in report.globals_:
        by_file.setdefault(hit.path, []).append(hit)

    edited = 0
    for path, hits in by_file.items():
        original = path.read_text(encoding="utf-8", errors="replace")
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")

        lines = original.splitlines(keepends=True)
        for hit in sorted(hits, key=lambda h: h.line_no, reverse=True):
            index = hit.line_no - 1
            raw = lines[index]
            indent = raw[: len(raw) - len(raw.lstrip())]
            newline = "\n" if raw.endswith("\n") else ""
            lines[index] = f"{indent}# {raw.strip()}  {MARKER}{newline}"
            edited += 1
        path.write_text("".join(lines), encoding="utf-8")
        print(f"  edited {path} ({len(hits)} line(s)); backup at {backup.name}")
    return edited


def print_report(report: Report) -> None:
    print(f"scanned {report.scanned} file(s)\n")

    if report.clean:
        print("CLEAN: no live sub-LM references in either the REPL globals or the "
              "system prompt.")
        return

    if report.globals_:
        print(f"REPL globals / definitions ({len(report.globals_)}):")
        for hit in report.globals_:
            print(f"  {hit.path}:{hit.line_no}: {hit.line.strip()[:110]}")
        print()

    if report.prompts:
        print(f"System prompt mentions ({len(report.prompts)}) -- FIX THESE BY HAND:")
        for hit in report.prompts:
            print(f"  {hit.path}:{hit.line_no}: {hit.line.strip()[:110]}")
        print(
            "  --apply will not touch these. Commenting out a line inside a\n"
            "  triple-quoted string does not remove it from the prompt; it puts a\n"
            "  '#' in the string and the model still reads llm_query. Rewrite the\n"
            "  wording so the prompt describes a REPL with no sub-LM available.\n"
        )

    if report.asymmetric:
        side = "prompt" if report.prompts else "globals"
        other = "globals" if report.prompts else "prompt"
        print(
            f"ASYMMETRIC: the {other} side is clean but the {side} side is not.\n"
            f"This is the worst of the three states. If the prompt still offers a\n"
            f"sub-LM the model has no way to call, every rollout dies on NameError\n"
            f"while the run looks perfectly healthy from outside -- busy GPUs,\n"
            f"advancing steps, a plausible loss curve, and a scorer quietly\n"
            f"returning zeros. Fix both sides before launching."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"rlm training source tree (default {DEFAULT_ROOT})")
    parser.add_argument("--apply", action="store_true",
                        help="comment out the references, then re-verify")
    parser.add_argument("--verify", action="store_true",
                        help="report only; change nothing (the default)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = scan(args.root)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.apply and report.globals_:
        print(f"editing {len(report.globals_)} code binding(s)...")
        apply(report)
        print("\nre-scanning to verify...\n")
        report = scan(args.root)
    elif args.apply and report.prompts:
        print("nothing to edit automatically: every hit is prompt text.\n")

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print_report(report)

    if not report.clean:
        print(
            "\nNot clean. Do not launch. Any remaining reference means V0 is testing "
            "a condition that includes recursion, which is a different experiment.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
