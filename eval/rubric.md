# Grading rubric

Every answer is graded against the reference answer with exactly one label.
The rubric was fixed before any answers were generated and wasn't changed afterwards.

| Label | Meaning |
|---|---|
| **CORRECT** | States the key facts of the reference with no material errors. Wording can differ. |
| **PARTIAL** | Some key facts are right, but something important is missing or wrong. |
| **INCORRECT** | Wrong, contradicts the reference, off-topic, or refuses when the reference has an answer. |

**Accuracy** = share of prompts graded CORRECT. PARTIAL counts as not correct.

Rules for graders:
- Grade from `results/grading_sheet.csv` only. It has no model names, and the two
  answers to each prompt are shuffled into random positions.
- Don't open `results/blind_key.json` until every row is graded.
- Ignore length, formatting and tone. A confident wrong answer is INCORRECT.
- When unsure between two labels, pick the lower one.
