"""Thin FastAPI service in front of vLLM.

    VLLM_URL=http://localhost:8000/v1 uvicorn serve.api:app --port 8080

Adds what clients need and vLLM doesn't do by itself: an API key check, input
limits, a fixed system prompt, timeouts, and a plain JSON response.
"""
from __future__ import annotations

import os
import time

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")
MODEL = os.getenv("SERVED_MODEL", "domain")
API_KEYS = {k for k in os.getenv("API_KEYS", "").split(",") if k}
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a domain expert. Answer accurately and concisely. "
                          "If the question can't be answered from what you know, say so instead of guessing.")
TIMEOUT_S = float(os.getenv("UPSTREAM_TIMEOUT_S", "60"))

app = FastAPI(title="Domain QA", version="1.0.0")
_client = httpx.AsyncClient(base_url=VLLM_URL, timeout=TIMEOUT_S)


def check_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEYS and x_api_key not in API_KEYS:
        raise HTTPException(401, "invalid API key")


class AnswerRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    max_tokens: int = Field(default=512, ge=1, le=1024)


class AnswerResponse(BaseModel):
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@app.get("/healthz")
async def healthz():
    try:
        r = await _client.get("/models", timeout=3)
        return {"ok": r.status_code == 200}
    except httpx.HTTPError:
        raise HTTPException(503, "model server unavailable") from None


@app.post("/v1/answer", response_model=AnswerResponse, dependencies=[Depends(check_key)])
async def answer(req: AnswerRequest):
    start = time.perf_counter()
    try:
        r = await _client.post("/chat/completions", json={
            "model": MODEL, "temperature": 0.0, "max_tokens": req.max_tokens,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": req.question}]})
    except httpx.TimeoutException:
        raise HTTPException(504, "model timed out") from None
    if r.status_code != 200:
        raise HTTPException(502, f"model server error {r.status_code}")
    data = r.json()
    usage = data.get("usage", {})
    return AnswerResponse(answer=data["choices"][0]["message"]["content"].strip(), model=MODEL,
                          prompt_tokens=usage.get("prompt_tokens", 0),
                          completion_tokens=usage.get("completion_tokens", 0),
                          latency_ms=round((time.perf_counter() - start) * 1000, 1))
