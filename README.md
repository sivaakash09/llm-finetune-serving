# LLM Fine-Tuning & Serving

QLoRA fine-tuning of **Qwen2.5-7B-Instruct** for domain question answering, a
**blinded** before/after evaluation, and production serving with **vLLM + FastAPI**
on an AWS GPU instance.

## Results

| | Base model | Fine-tuned | Change |
|---|---|---|---|
| **Accuracy** (300 held-out prompts, blind rubric grading) | 68% | **80%** | +12 points |

| Serving (same GPU, same load) | Output tokens/sec | Change |
|---|---|---|
| Hugging Face `generate()` (baseline) | 22 | |
| vLLM | **33** | +50% |

Both servers were benchmarked on the same instance with the same prompts,
prompt length, output length and concurrency (`bench/throughput.py`).

## What's here

```
configs/qlora_qwen7b.yaml     all training settings in one place
src/ftserve/
  data.py                     JSONL -> chat format; loss only on answer tokens
  train_qlora.py              4-bit NF4 base + LoRA (r=16) with early stopping
  generate.py                 base vs. fine-tuned answers on the held-out prompts (vLLM + LoRA)
  blind_eval.py               blind -> grade -> unblind, with a bootstrap 95% CI
eval/rubric.md                the fixed grading rubric
serve/
  start_vllm.sh               vLLM OpenAI-compatible server with the adapter
  api.py                      FastAPI front end: API keys, limits, timeouts
  naive_server.py             plain HF baseline with the same API, for the benchmark
bench/throughput.py           fixed-load throughput and latency benchmark
scripts/make_splits.py        stable, leak-free train / val / eval split
colab/RUN_IN_COLAB.md         every step as copy-paste Colab cells
tests/                        pytest for data masking, splitting, blinding, stats
```

## Training

- **Base:** Qwen2.5-7B-Instruct loaded in **4-bit NF4** with double quantization
  (bitsandbytes), so the 7B model fits on one 24 GB GPU.
- **Adapters:** LoRA r=16, alpha=32, dropout 0.05 on all attention and MLP
  projections. About 40M trainable parameters, roughly 0.5% of the model.
- **Loss masking:** prompt tokens are labelled `-100`, so the model learns to
  produce answers, not to repeat questions. The code checks that the chat
  template keeps the prompt as an exact prefix before masking.
- **Optimiser:** paged 8-bit AdamW, lr 2e-4 with cosine decay, effective batch 16,
  gradient checkpointing.
- **Model selection:** evaluated every 50 steps on a separate validation split;
  the best checkpoint by validation loss is kept, with early stopping.

## Evaluation

The question was simple: does the fine-tuned model answer more domain questions
correctly than the base model? To keep it honest:

1. **Held-out prompts.** 300 prompts split off by a hash of their id before
   training (`scripts/make_splits.py`). Near-duplicate questions are removed
   from training so nothing leaks.
2. **Same conditions.** Both models answer with the same system prompt and
   greedy decoding; the only difference is the adapter.
3. **Blind grading.** Answers are shuffled into A/B slots per prompt and mixed
   across the sheet. Graders see no model names. The key is opened only after
   grading.
4. **Fixed rubric** (`eval/rubric.md`), written before generating anything.
   Only CORRECT counts; PARTIAL counts as wrong.
5. **Uncertainty.** `unblind` reports a paired bootstrap 95% interval for the
   difference, plus how many prompts only one model got right.

## Serving

- **vLLM** serves the base model with the LoRA adapter attached
  (`serve/start_vllm.sh`), using continuous batching, paged KV cache and prefix caching.
- **FastAPI** (`serve/api.py`) sits in front: API-key check, input size limits,
  a fixed system prompt, upstream timeouts, and a clean JSON response.
- Deployed on an AWS GPU instance. The settings target a single 24 GB GPU, such as the A10G on a g5.xlarge.

```bash
pip install -r requirements-serve.txt
./serve/start_vllm.sh outputs/qwen7b-domain-qlora/adapter          # port 8000
API_KEYS=change-me uvicorn serve.api:app --host 0.0.0.0 --port 8080

curl -s localhost:8080/v1/answer -H 'x-api-key: change-me' -H 'content-type: application/json' \
  -d '{"question": "How is working capital calculated?"}'
```

## Benchmark method

`bench/throughput.py` sends the same fixed workload to any OpenAI-compatible
server: 32 requests of a ~512-token prompt, exactly 256 output tokens each
(`ignore_eos`), after 4 warm-up requests. It reports output tokens per second
over the whole run and p50/p95 latency.

The baseline (`serve/naive_server.py`) exposes the same API using plain
`transformers.generate()` in fp16 with the adapter unmerged. That's what a
straightforward deployment looks like before any serving work.

## Reproducing

Every step, in order, as Colab cells: **[colab/RUN_IN_COLAB.md](colab/RUN_IN_COLAB.md)**.

The training data isn't included. `data/sample/sample.jsonl` shows the format:

```json
{"id": "ex-002", "question": "How is working capital calculated?", "answer": "Working capital is current assets minus current liabilities."}
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

They run on CPU and cover answer-only loss masking, dropping over-long
examples, the leak-free split, blinding and unblinding, label parsing and the
bootstrap interval.
