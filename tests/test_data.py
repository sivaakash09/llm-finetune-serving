import json

import pytest

from ftserve.data import IGNORE_INDEX, build_messages, load_jsonl, tokenize_all, tokenize_example
from scripts.make_splits import split


class FakeTokenizer:
    """Character-level stand-in with a chat template shaped like ChatML."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        if messages[-1]["role"] == "assistant":
            # same prefix as the generation prompt, then the answer
            text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages[:-1])
            text += f"<assistant>{messages[-1]['content']}</assistant>"
        return [ord(c) for c in text]


ROW = {"id": "a", "question": "What is working capital?", "answer": "Current assets minus current liabilities."}


def test_only_answer_tokens_are_trained():
    ex = tokenize_example(ROW, FakeTokenizer(), "sys", max_len=10_000)
    trained = "".join(chr(t) for t, lab in zip(ex["input_ids"], ex["labels"]) if lab != IGNORE_INDEX)
    assert trained == ROW["answer"] + "</assistant>"
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])


def test_too_long_prompts_are_dropped():
    out, dropped = tokenize_all([ROW], FakeTokenizer(), "sys", max_len=10)
    assert out == [] and dropped == 1


def test_context_goes_into_user_turn():
    msgs = build_messages({**ROW, "context": "Ref text"}, "sys", include_answer=False)
    assert msgs[1]["content"].startswith("Reference:\nRef text")
    assert len(msgs) == 2


def test_load_jsonl_validates(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"question": "q", "answer": ""}) + "\n")
    with pytest.raises(ValueError):
        load_jsonl(p)


def test_split_is_stable_and_leak_free():
    rows = [{"id": f"r{i}", "question": f"question {i}", "answer": "a"} for i in range(100)]
    rows.append({"id": "dup", "question": "Question 3!", "answer": "a"})  # near-duplicate of r3
    t1, v1, e1 = split(rows, 20, 10)
    t2, v2, e2 = split(list(reversed(rows)), 20, 10)
    assert [r["id"] for r in e1] == [r["id"] for r in e2]
    eval_q = {r["question"].lower().strip("!") for r in e1}
    assert not any(r["question"].lower().strip("!") in eval_q for r in t1)
    assert not {r["id"] for r in t1} & {r["id"] for r in e1}
