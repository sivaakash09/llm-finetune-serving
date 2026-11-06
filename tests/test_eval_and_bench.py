import pytest

from bench.throughput import make_prompt, percentile
from ftserve.blind_eval import blind, bootstrap_diff_ci, parse_label, unblind

GENS = [{"id": f"p{i}", "question": f"q{i}", "reference": "ref", "base": f"base-{i}", "tuned": f"tuned-{i}"}
        for i in range(40)]


def test_sheet_hides_model_identity():
    sheet, key = blind(GENS, seed=7)
    assert len(sheet) == 80
    assert set(sheet[0]) == {"item", "question", "reference", "answer", "grade"}
    # both slots used for both models (not always base in A)
    assert {v["A"] for v in key.values()} == {"base", "tuned"}


def test_blinding_is_reproducible():
    assert blind(GENS, seed=7)[1] == blind(GENS, seed=7)[1]


def test_unblind_recovers_true_accuracy():
    sheet, key = blind(GENS, seed=3)
    # grade: tuned correct on all, base correct on the first 10 prompts
    for row in sheet:
        pid, slot = row["item"].rsplit(":", 1)
        model = key[pid][slot]
        ok = model == "tuned" or int(pid[1:]) < 10
        row["grade"] = "CORRECT" if ok else "PARTIAL"
    rep = unblind(sheet, key)
    assert rep["accuracy"] == {"base": 0.25, "tuned": 1.0}
    assert rep["difference_points"] == 75.0
    assert rep["only_tuned_correct"] == 30 and rep["only_base_correct"] == 0
    lo, hi = rep["difference_95ci_points"]
    assert lo <= 75.0 <= hi


def test_partial_counts_as_wrong():
    sheet, key = blind(GENS[:1], seed=1)
    for row in sheet:
        row["grade"] = "PARTIAL"
    assert unblind(sheet, key)["accuracy"] == {"base": 0.0, "tuned": 0.0}


def test_parse_label():
    assert parse_label("correct") == "CORRECT"
    assert parse_label("Label: INCORRECT.") == "INCORRECT"
    with pytest.raises(ValueError):
        parse_label("looks good")


def test_bootstrap_ci_zero_when_identical():
    lo, hi = bootstrap_diff_ci([1, 0, 1, 1], [1, 0, 1, 1], n=500)
    assert lo == hi == 0


def test_percentile_and_prompt():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 100) == 5
    assert len(make_prompt(380).split()) > 380
