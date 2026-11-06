#!/usr/bin/env bash
# Serve the base model with the LoRA adapter through vLLM's OpenAI-compatible server.
#   ./serve/start_vllm.sh outputs/qwen7b-domain-qlora/adapter
#
# Requests use model name "domain" for the fine-tuned model, or the base model
# name for the original. Throughput comes from continuous batching and paged KV
# cache; the flags below are the ones that mattered on a single 24 GB GPU.
set -euo pipefail
ADAPTER="${1:?usage: start_vllm.sh <adapter_dir>}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"

exec vllm serve "${BASE_MODEL}" \
  --enable-lora \
  --lora-modules domain="${ADAPTER}" \
  --max-lora-rank 16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --port "${PORT}"
