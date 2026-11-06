"""Blinded rubric evaluation: base vs. fine-tuned.

Three steps, so the grader never knows which model wrote which answer:

  1. blind   - for every prompt, shuffle the two answers into slots A/B with a
               seeded coin flip. Write the grading sheet (no model names) and,
               separately, the key that maps A/B back to base/tuned.
  2. grade   - each answer is scored against the reference with a fixed rubric
               (rubric.md). Grading is done by people in the CSV, or by a judge
               model through any OpenAI-compatible API (`--judge`).
  3. unblind - join grades with the key, report accuracy for each model, the
               paired difference and a bootstrap 95% confidence interval.

    python -m ftserve.blind_eval blind   --gens results/generations.jsonl
    python -m ftserve.blind_eval grade   --judge --base-url ... --model ...
    python -m ftserve.blind_eval unblind
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

RUBRIC = """Score the ANSWER against the REFERENCE using exactly one label:
CORRECT    - states the key facts of the reference with no material errors. Wording may differ.
PARTIAL    - some key facts right, but something important is missing or wrong.
INCORRECT  - wrong, contradicts the reference, off-topic, or refuses when the reference has an answer.
Only CORRECT counts toward accuracy. Ignore length and style. Do not reward confident tone.
Reply with the label only."""

LABELS = ("CORRECT", "PARTIAL", "INCORRECT")
SHEET = Path("results/grading_sheet.csv")
KEY = Path("results/blind_key.json")
REPORT = Path("results/eval_report.json")


def blind(gens: list[dict], seed: int = 1234) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    sheet, key = [], {}
    for g in gens:
        swap = rng.random() < 0.5
        a, b = (g["tuned"], g["base"]) if swap else (g["base"], g["tuned"])
        key[g["id"]] = {"A": "tuned" if swap else "base", "B": "base" if swap else "tuned"}
        for slot, ans in (("A", a), ("B", b)):
            sheet.append({"item": f"{g['id']}:{slot}", "question": g["question"], "reference": g["reference"],
                          "answer": ans, "grade": ""})
    rng.shuffle(sheet)  # so A/B of the same prompt aren't adjacent either
    return sheet, key


_LABEL = re.compile(r"\b(CORRECT|PARTIAL|INCORRECT)\b")


def parse_label(text: str) -> str:
    m = _LABEL.search(text.upper())
    if not m:
        raise ValueError(f"no rubric label in: {text[:80]!r}")
    return m.group(1)


def unblind(sheet: list[dict], key: dict) -> dict:
    per_model: dict[str, dict[str, int]] = {"base": {}, "tuned": {}}
    ungraded = 0
    for row in sheet:
        pid, slot = row["item"].rsplit(":", 1)
        label = (row.get("grade") or "").strip().upper()
        if label not in LABELS:
            ungraded += 1
            continue
        per_model[key[pid][slot]][pid] = 1 if label == "CORRECT" else 0
    ids = sorted(set(per_model["base"]) & set(per_model["tuned"]))
    if not ids:
        raise ValueError("no prompt has grades for both models")
    base = [per_model["base"][i] for i in ids]
    tuned = [per_model["tuned"][i] for i in ids]
    lo, hi = bootstrap_diff_ci(base, tuned)
    return {
        "n_prompts": len(ids),
        "ungraded_answers": ungraded,
        "accuracy": {"base": round(sum(base) / len(ids), 4), "tuned": round(sum(tuned) / len(ids), 4)},
        "difference_points": round(100 * (sum(tuned) - sum(base)) / len(ids), 1),
        "difference_95ci_points": [round(100 * lo, 1), round(100 * hi, 1)],
        "only_tuned_correct": sum(1 for b, t in zip(base, tuned) if t and not b),
        "only_base_correct": sum(1 for b, t in zip(base, tuned) if b and not t),
    }


def bootstrap_diff_ci(base: list[int], tuned: list[int], n: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% CI of accuracy(tuned) - accuracy(base), resampling prompts (paired)."""
    rng = random.Random(seed)
    k = len(base)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        diffs.append(sum(tuned[i] - base[i] for i in idx) / k)
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


# ------------------------------------------------------------------ CLI
def _read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def _write_sheet(rows):
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    with open(SHEET, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item", "question", "reference", "answer", "grade"])
        w.writeheader()
        w.writerows(rows)


def _read_sheet():
    with open(SHEET, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def judge_sheet(base_url: str, api_key: str, model: str) -> None:
    import httpx

    rows = _read_sheet()
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=60,
                      headers={"Authorization": f"Bearer {api_key}"}) as http:
        for n, row in enumerate(rows, 1):
            if row["grade"]:
                continue
            msg = (f"QUESTION:\n{row['question']}\n\nREFERENCE:\n{row['reference']}\n\n"
                   f"ANSWER:\n{row['answer']}")
            r = http.post("/chat/completions", json={
                "model": model, "temperature": 0, "max_tokens": 5,
                "messages": [{"role": "system", "content": RUBRIC}, {"role": "user", "content": msg}]})
            r.raise_for_status()
            row["grade"] = parse_label(r.json()["choices"][0]["message"]["content"])
            if n % 25 == 0:
                print(f"[grade] {n}/{len(rows)}", flush=True)
                _write_sheet(rows)  # checkpoint, so a crash doesn't lose work
    _write_sheet(rows)
    print("[grade] DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("blind")
    b.add_argument("--gens", default="results/generations.jsonl")
    b.add_argument("--seed", type=int, default=1234)
    g = sub.add_parser("grade")
    g.add_argument("--judge", action="store_true", help="grade with a judge model instead of by hand")
    g.add_argument("--base-url")
    g.add_argument("--api-key", default="none")
    g.add_argument("--model")
    sub.add_parser("unblind")
    args = ap.parse_args()

    if args.cmd == "blind":
        sheet, key = blind(_read_jsonl(args.gens), seed=args.seed)
        _write_sheet(sheet)
        KEY.write_text(json.dumps(key, indent=2))
        print(f"[blind] {len(sheet)} answers to grade in {SHEET}; key saved to {KEY} (don't open it until done)")
    elif args.cmd == "grade":
        if not args.judge:
            print(f"Fill the 'grade' column of {SHEET} with CORRECT / PARTIAL / INCORRECT using rubric.md.")
            return
        judge_sheet(args.base_url, args.api_key, args.model)
    else:
        report = unblind(_read_sheet(), json.loads(KEY.read_text()))
        REPORT.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
