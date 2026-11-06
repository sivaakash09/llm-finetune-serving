"""Generate answers to the held-out prompts with the base model and with the adapter.

    python -m ftserve.generate --config configs/qlora_qwen7b.yaml \
        --prompts data/eval_prompts.jsonl --adapter outputs/qwen7b-domain-qlora/adapter \
        --out results/generations.jsonl

Both systems use the same prompts, system prompt, decoding settings and seed, so
the only difference is the adapter. Uses vLLM's offline engine with LoRA for speed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ftserve.data import build_messages, load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", required=True, help="JSONL with id, question, answer (reference)")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="results/generations.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    cfg = yaml.safe_load(Path(args.config).read_text())
    sys_prompt = cfg["data"]["system_prompt"].strip()
    rows = load_jsonl(args.prompts)
    print(f"[generate] {len(rows)} prompts", flush=True)

    llm = LLM(model=cfg["base_model"], enable_lora=True, max_lora_rank=cfg["lora"]["r"],
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    convs = [build_messages(r, sys_prompt, include_answer=False) for r in rows]

    print("[generate] base model...", flush=True)
    base = llm.chat(convs, params, use_tqdm=True)
    print("[generate] base + adapter...", flush=True)
    tuned = llm.chat(convs, params, use_tqdm=True, lora_request=LoRARequest("domain", 1, args.adapter))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r, b, t in zip(rows, base, tuned):
            f.write(json.dumps({"id": r["id"], "question": r["question"], "reference": r["answer"],
                                "base": b.outputs[0].text.strip(), "tuned": t.outputs[0].text.strip()}) + "\n")
    print(f"[generate] wrote {args.out}", flush=True)
    print("[generate] DONE", flush=True)


if __name__ == "__main__":
    main()
