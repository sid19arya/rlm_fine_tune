#!/usr/bin/env python3
"""Append to `interventions.jsonl` -- the log of every time a human had to act.

Part of the point of this run is measuring how far an agent gets unaided, so the
log is the experiment, not a scorecard. Omitting an entry to look better
destroys the only data the run produces about its own execution.

    python interventions.py --task 3 --category DEPS \
      --tried "uv pip install -e . failed building flash-attn" \
      --human-did "installed the prebuilt wheel"

    python interventions.py --summary
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path("interventions.jsonl")

CATEGORIES = {
    "AUTH": "credentials, keys, scopes",
    "PROVISION": "getting hardware at all",
    "ORDERING": "steps run in the wrong order, or a missing prerequisite",
    "DEPS": "installs, builds, version conflicts",
    "CONFIG": "a setting that had to change",
    "LONGRUN": "anything only visible over hours",
    "DIAGNOSE": "the agent could not work out what was wrong",
}


def append(path: Path, entry: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarise(rows: list[dict]) -> str:
    if not rows:
        return "no interventions logged (yet -- check that entries are being written)"

    by_category = Counter(r.get("category", "?") for r in rows)
    by_task = Counter(str(r.get("task", "?")) for r in rows)
    unrecoverable = [r for r in rows if not r.get("recoverable", True)]

    lines = [f"{len(rows)} intervention(s)", "", "by category:"]
    lines += [f"  {name:<10} {count}" for name, count in by_category.most_common()]
    lines += ["", "by task:"]
    lines += [f"  task {task:<5} {count}" for task, count in sorted(by_task.items())]
    if unrecoverable:
        lines += ["", f"{len(unrecoverable)} the agent could not have recovered from:"]
        lines += [f"  task {r.get('task')}: {r.get('human_did', '')[:80]}"
                  for r in unrecoverable]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--task", type=int)
    parser.add_argument("--category", choices=sorted(CATEGORIES))
    parser.add_argument("--tried", help="what the agent attempted, twice")
    parser.add_argument("--human-did", help="what the human actually did")
    parser.add_argument("--unrecoverable", action="store_true",
                        help="the agent could not have got past this alone")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)

    if args.summary:
        print(summarise(read(args.path)))
        return 0

    missing = [name for name, value in
               (("--task", args.task), ("--category", args.category),
                ("--tried", args.tried), ("--human-did", args.human_did))
               if value in (None, "")]
    if missing:
        parser.error(f"missing {', '.join(missing)}. Categories: "
                     + "; ".join(f"{k} ({v})" for k, v in CATEGORIES.items()))

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "category": args.category,
        "tried": args.tried,
        "human_did": args.human_did,
        "recoverable": not args.unrecoverable,
    }
    append(args.path, entry)
    print(f"logged: task {entry['task']} [{entry['category']}] -> {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
