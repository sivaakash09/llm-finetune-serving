"""Training data: JSONL -> chat-formatted, tokenized examples with loss only on the answer.

Input format, one example per line:

    {"id": "q-0001", "question": "...", "answer": "...", "context": "optional reference text"}

The model is trained to produce `answer` given the system prompt and question.
Prompt tokens get label -100, so the loss (and therefore the learning) comes
only from the answer. Without this the model spends capacity memorising questions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

IGNORE_INDEX = -100


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("question", "answer"):
                if not str(row.get(key, "")).strip():
                    raise ValueError(f"{path}:{n}: missing '{key}'")
            row.setdefault("id", f"row-{n}")
            rows.append(row)
    return rows


def build_messages(row: dict, system_prompt: str, include_answer: bool = True) -> list[dict]:
    user = row["question"]
    if row.get("context"):
        user = f"Reference:\n{row['context']}\n\nQuestion: {row['question']}"
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
    if include_answer:
        msgs.append({"role": "assistant", "content": row["answer"]})
    return msgs


def tokenize_example(row: dict, tokenizer: Any, system_prompt: str, max_len: int) -> dict | None:
    """Tokenize one example and mask everything except the assistant answer.

    Returns None if the prompt alone doesn't fit, so the caller can count and drop it.
    """
    prompt_ids = tokenizer.apply_chat_template(
        build_messages(row, system_prompt, include_answer=False), tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(build_messages(row, system_prompt), tokenize=True)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template changed the prompt when the answer was appended; can't mask safely")
    if len(prompt_ids) >= max_len:
        return None
    full_ids = full_ids[:max_len]
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def tokenize_all(rows: list[dict], tokenizer: Any, system_prompt: str, max_len: int) -> tuple[list[dict], int]:
    out, dropped = [], 0
    for r in rows:
        ex = tokenize_example(r, tokenizer, system_prompt, max_len)
        if ex is None:
            dropped += 1
        else:
            out.append(ex)
    return out, dropped


class PadCollator:
    """Right-pad a batch; labels are padded with IGNORE_INDEX."""

    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        width = max(len(b["input_ids"]) for b in batch)

        def pad(seq, value):
            return seq + [value] * (width - len(seq))

        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], IGNORE_INDEX) for b in batch]),
        }
