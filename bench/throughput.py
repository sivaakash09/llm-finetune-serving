"""Output-token throughput benchmark for any OpenAI-compatible server.

    python bench/throughput.py --url http://localhost:8000/v1 --label vllm
    python bench/throughput.py --url http://localhost:8001/v1 --label naive

Both runs use the same fixed load so the numbers are comparable: the same
number of requests, the same concurrency, the same prompt length and the same
output length (ignore_eos forces every reply to exactly --output-tokens).

Reports output tokens/second across the whole run, plus p50/p95 request latency.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

# ~512-token prompt built from a fixed passage, so every request is identical in size
PASSAGE = ("The quarterly report covers revenue, operating costs, headcount, and risk factors "
           "across all business units, with notes on assumptions and data sources. ")


def make_prompt(target_words: int) -> str:
    words = PASSAGE.split()
    text = " ".join(words[i % len(words)] for i in range(target_words))
    return f"Summarise the following text in detail.\n\n{text}"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def run(url: str, model: str, n: int, concurrency: int, prompt_words: int, output_tokens: int,
              warmup: int = 4) -> dict:
    prompt = make_prompt(prompt_words)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_tokens, "temperature": 0.0, "ignore_eos": True}
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    tokens = 0
    errors = 0

    async with httpx.AsyncClient(base_url=url.rstrip("/"), timeout=600) as client:
        for _ in range(warmup):  # load kernels / caches before timing
            await client.post("/chat/completions", json=body)

        async def one(i: int):
            nonlocal tokens, errors
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post("/chat/completions", json=body)
                    r.raise_for_status()
                    tokens += r.json()["usage"]["completion_tokens"]
                    latencies.append(time.perf_counter() - t0)
                except httpx.HTTPError:
                    errors += 1
                done = len(latencies) + errors
                if done % max(n // 10, 1) == 0:
                    print(f"[bench] {done}/{n} requests", flush=True)

        start = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.perf_counter() - start

    return {"requests": n, "errors": errors, "concurrency": concurrency, "prompt_words": prompt_words,
            "output_tokens_per_request": output_tokens, "wall_seconds": round(wall, 2),
            "output_tokens_per_second": round(tokens / wall, 1),
            "latency_p50_s": round(statistics.median(latencies), 2) if latencies else None,
            "latency_p95_s": round(percentile(latencies, 95), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="domain")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=1)   # single stream, same for both servers
    ap.add_argument("--prompt-words", type=int, default=380)   # ~512 tokens
    ap.add_argument("--output-tokens", type=int, default=256)
    args = ap.parse_args()

    res = asyncio.run(run(args.url, args.model, args.requests, args.concurrency, args.prompt_words,
                          args.output_tokens))
    res["label"] = args.label
    out = Path(f"results/bench_{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("[bench] DONE", flush=True)


if __name__ == "__main__":
    main()
