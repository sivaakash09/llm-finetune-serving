"""Split one JSONL dataset into train / val / held-out eval prompts.

    python scripts/make_splits.py --data data/private/all.jsonl --eval-size 300 --val-size 200

The split is by a hash of each example's id, so it's stable across runs and an
example can never move between train and eval. Near-duplicate questions are
removed from train if they also appear in eval, so the eval set stays unseen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def norm(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


def bucket(example_id: str) -> float:
    return int(hashlib.sha256(example_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def split(rows: list[dict], eval_size: int, val_size: int) -> tuple[list[dict], list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda r: bucket(str(r["id"])))
    eval_rows = ordered[:eval_size]
    val_rows = ordered[eval_size : eval_size + val_size]
    seen = {norm(r["question"]) for r in eval_rows}
    train = [r for r in ordered[eval_size + val_size :] if norm(r["question"]) not in seen]
    return train, val_rows, eval_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-size", type=int, default=300)
    ap.add_argument("--val-size", type=int, default=200)
    ap.add_argument("--out-dir", default="data")
    a = ap.parse_args()

    rows = [json.loads(x) for x in Path(a.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    for i, r in enumerate(rows):
        r.setdefault("id", f"row-{i}")
    train, val, ev = split(rows, a.eval_size, a.val_size)
    out = Path(a.out_dir)
    for name, part in (("train.jsonl", train), ("val.jsonl", val), ("eval_prompts.jsonl", ev)):
        (out / name).write_text("".join(json.dumps(r) + "\n" for r in part), encoding="utf-8")
        print(f"[split] {name}: {len(part)}")


if __name__ == "__main__":
    main()
