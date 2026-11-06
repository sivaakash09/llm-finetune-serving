# Running everything in Google Colab

Use an **L4** or **A100** runtime (Runtime → Change runtime type). A T4 also works
for training if you set `max_seq_len: 1024` and `compute_dtype: float16` in the config.

Run the cells **in order**. Each long step prints progress lines and ends with a
`... DONE` line. If you don't see `DONE`, it's still running (or it crashed: the
last lines of the output will show the error).

**Order:** 1 setup → 2 data → 3 train → 4 generate → 5 grade → 6 unblind → 7 benchmark.
Steps 4 to 6 need step 3's adapter. Step 7 needs it too, but is independent of 4 to 6.

---

### Cell 1: setup (about 3 minutes)

```python
!git clone https://github.com/sivaakash09/llm-finetune-serving.git
%cd llm-finetune-serving
!pip -q install -r requirements.txt
!pip -q install -r requirements-serve.txt
import os; os.environ["PYTHONPATH"] = "src:."
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

### Cell 2: data

Upload your dataset (one JSONL, format in `data/sample/sample.jsonl`) to
`data/private/all.jsonl`, then split it:

```python
!mkdir -p data/private
# from google.colab import files; files.upload()  # then move the file to data/private/all.jsonl
!python scripts/make_splits.py --data data/private/all.jsonl --eval-size 300 --val-size 200
```

You should see three `[split]` lines with counts.

### Cell 3: train (about 1 to 3 hours depending on GPU and data size)

```python
!python -m ftserve.train_qlora --config configs/qlora_qwen7b.yaml 2>&1 | tee train.log
```

Prints `[train] step X/Y ... eta Nm` every 10 steps and ends with `[train] DONE`.
Quick sanity run first if you want: add `--max-steps 20`.

If the Colab output stops updating, check from a second cell:

```python
!tail -n 3 train.log
!nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

GPU utilisation above 0% means it's still training.

### Cell 4: generate answers, base vs. fine-tuned (about 10 minutes)

```python
!python -m ftserve.generate --config configs/qlora_qwen7b.yaml \
    --prompts data/eval_prompts.jsonl \
    --adapter outputs/qwen7b-domain-qlora/adapter \
    --out results/generations.jsonl
```

Ends with `[generate] DONE`.

### Cell 5: blind and grade

```python
!python -m ftserve.blind_eval blind --gens results/generations.jsonl
```

Then **either** grade by hand (download `results/grading_sheet.csv`, fill the
`grade` column using `eval/rubric.md`, upload it back to the same path), **or**
use a judge model through any OpenAI-compatible API:

```python
!python -m ftserve.blind_eval grade --judge --base-url https://api.openai.com/v1 \
    --api-key "$JUDGE_API_KEY" --model gpt-4o
```

Prints `[grade] N/600` every 25 answers and `[grade] DONE` at the end.

### Cell 6: unblind and report

```python
!python -m ftserve.blind_eval unblind
```

Prints base vs. tuned accuracy, the difference in points and its 95% interval.
Saved to `results/eval_report.json`.

### Cell 7: throughput benchmark (about 15 minutes)

Free GPU memory from earlier cells first (Runtime → Restart session, then re-run
Cell 1's `%cd` and `os.environ` lines).

```python
# vLLM server in the background
!nohup ./serve/start_vllm.sh outputs/qwen7b-domain-qlora/adapter > vllm.log 2>&1 &
!for i in $(seq 1 60); do curl -s localhost:8000/v1/models >/dev/null && echo up && break; sleep 5; done
!python bench/throughput.py --url http://localhost:8000/v1 --label vllm
!pkill -f "vllm serve"; sleep 10
```

```python
# naive Hugging Face server, same hardware and same load
!nohup python serve/naive_server.py --adapter outputs/qwen7b-domain-qlora/adapter --port 8001 > naive.log 2>&1 &
!for i in $(seq 1 60); do curl -s localhost:8001/v1/models >/dev/null && echo up && break; sleep 5; done
!python bench/throughput.py --url http://localhost:8001/v1 --label naive
!pkill -f naive_server.py
```

Each benchmark ends with `[bench] DONE` and writes `results/bench_<label>.json`.
If a server never prints `up`, look at the end of `vllm.log` or `naive.log`.
