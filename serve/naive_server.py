"""Baseline server for the throughput comparison: plain Hugging Face `generate()`.

It exposes the same OpenAI-style /v1/chat/completions route as vLLM so the
benchmark hits both servers the same way. Requests are handled one at a time
(a lock around generate), which is how a straightforward HF deployment behaves:
no continuous batching, no paged KV cache.

    python serve/naive_server.py --base Qwen/Qwen2.5-7B-Instruct --adapter outputs/.../adapter --port 8001
"""
from __future__ import annotations

import argparse
import threading
import time
import uuid

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class ChatRequest(BaseModel):
    model: str = "domain"
    messages: list[dict]
    max_tokens: int = 256
    temperature: float = 0.0
    ignore_eos: bool = False


def build_app(base: str, adapter: str | None) -> FastAPI:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="auto")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    lock = threading.Lock()
    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": "domain"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        ids = tok.apply_chat_template(req.messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        kwargs = {"max_new_tokens": req.max_tokens, "do_sample": req.temperature > 0}
        if req.ignore_eos:
            kwargs["min_new_tokens"] = req.max_tokens
        with lock, torch.inference_mode():
            out = model.generate(ids, **kwargs)
        new = out[0, ids.shape[1]:]
        return {
            "id": uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()), "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant",
                                                 "content": tok.decode(new, skip_special_tokens=True)},
                         "finish_reason": "length"}],
            "usage": {"prompt_tokens": int(ids.shape[1]), "completion_tokens": int(new.shape[0])},
        }

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter")
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    uvicorn.run(build_app(a.base, a.adapter), host="0.0.0.0", port=a.port)
