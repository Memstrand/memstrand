"""memstrand CLI."""

from __future__ import annotations

import argparse

from .api import Memory
from .schema import Category, Provenance


def main() -> None:
    p = argparse.ArgumentParser(prog="memstrand", description="local-first agent memory")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("remember", help="store one memory")
    r.add_argument("content")
    r.add_argument("--category", default="fact", choices=[c.value for c in Category])
    r.add_argument("--task", default=None)

    q = sub.add_parser("recall", help="scoped recall")
    q.add_argument("query")
    q.add_argument("--task", default=None)
    q.add_argument("--limit", type=int, default=10)

    args = p.parse_args()
    mem = Memory()

    if args.cmd == "remember":
        atom = mem.remember(
            args.content,
            category=Category(args.category),
            provenance=Provenance.explicit_statement,
            task_id=args.task,
        )
        print(atom.id)
    elif args.cmd == "recall":
        for hit in mem.recall(args.query, task_id=args.task, limit=args.limit):
            print(f"{hit['score']:.3f}  {hit['id']}")
