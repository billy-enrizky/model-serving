"""OpenAI-compatible HTTP API for the Gemma 4 MTP engine.

Endpoints:
  GET  /healthz
  GET  /metrics                   Prometheus, no auth
  GET  /v1/models                 auth required
  POST /v1/chat/completions       auth required (stream + non-stream)

Auth: X-API-Key header, or Authorization: Bearer <key>.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterator

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from server.mtp_engine import GenerationStats, MTPEngine, get_engine

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

API_KEY = os.getenv("MODEL_API_KEY")
if not API_KEY:
    raise RuntimeError("MODEL_API_KEY env var required")

SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "gemma-4-E2B-it")


# Prometheus metrics
REQ_LATENCY = Histogram(
    "mtp_request_latency_seconds",
    "End-to-end request latency",
    ["endpoint", "status"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)
TTFT_HIST = Histogram(
    "mtp_ttft_seconds",
    "Time to first token (streaming)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
DECODE_TPS = Histogram(
    "mtp_decode_tokens_per_second",
    "Per-request decode throughput (tokens/sec)",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
)
ACCEPT_RATE = Histogram(
    "mtp_acceptance_rate",
    "Speculative decoding acceptance rate per request",
    buckets=(0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
COMPLETION_TOKENS = Counter(
    "mtp_completion_tokens_total", "Total completion tokens generated"
)
ACCEPTED_TOKENS = Counter("mtp_accepted_tokens_total", "Total accepted speculative tokens")
PROPOSED_TOKENS = Counter("mtp_proposed_tokens_total", "Total proposed speculative tokens")
AUTH_FAIL = Counter("mtp_auth_failures_total", "Auth failures")
VRAM_USED = Gauge("mtp_vram_used_bytes", "VRAM used (bytes), per device", ["device"])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("warming up engine")
    app.state.engine = get_engine()
    logger.info(
        "engine ready: target=%s assistant=%s",
        app.state.engine.target_model_id,
        app.state.engine.assistant_model_id,
    )
    yield


app = FastAPI(title="gemma-4-mtp-server", lifespan=lifespan)


def _check_auth(request: Request) -> None:
    presented = request.headers.get("x-api-key")
    if not presented:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            presented = auth.split(" ", 1)[1].strip()
    if not presented or not secrets.compare_digest(presented, API_KEY):
        AUTH_FAIL.inc()
        raise HTTPException(status_code=401, detail="invalid api key")


def _record_vram() -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            VRAM_USED.labels(device=f"cuda:{i}").set(torch.cuda.memory_allocated(i))


# ---------------------------------------------------------------------------
# Schemas (OpenAI-compatible subset)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = SERVED_MODEL_NAME
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    stream: bool = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz(request: Request) -> JSONResponse:
    eng: MTPEngine = request.app.state.engine
    return JSONResponse(
        {
            "status": "ok",
            "target_model": eng.target_model_id,
            "assistant_model": eng.assistant_model_id,
            "cuda": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    )


@app.get("/metrics")
async def metrics() -> Response:
    _record_vram()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    _check_auth(request)
    eng: MTPEngine = request.app.state.engine
    now = int(time.time())
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": SERVED_MODEL_NAME,
                    "object": "model",
                    "created": now,
                    "owned_by": "google",
                    "metadata": {
                        "target_model": eng.target_model_id,
                        "assistant_model": eng.assistant_model_id,
                    },
                }
            ],
        }
    )


def _record_stats(endpoint: str, status: str, stats: GenerationStats, total_time: float) -> None:
    REQ_LATENCY.labels(endpoint, status).observe(total_time)
    if stats.completion_tokens:
        decode_tps = stats.completion_tokens / max(stats.total_seconds - stats.ttft_seconds, 1e-6)
        DECODE_TPS.observe(decode_tps)
        COMPLETION_TOKENS.inc(stats.completion_tokens)
    if stats.proposed_tokens:
        ACCEPT_RATE.observe(stats.acceptance_rate)
        ACCEPTED_TOKENS.inc(stats.accepted_tokens)
        PROPOSED_TOKENS.inc(stats.proposed_tokens)
    if stats.ttft_seconds > 0:
        TTFT_HIST.observe(stats.ttft_seconds)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request) -> Response:
    _check_auth(request)
    eng: MTPEngine = request.app.state.engine
    messages = [m.model_dump() for m in req.messages]
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if not req.stream:
        wall_start = time.perf_counter()
        try:
            text, stats = eng.generate(
                messages=messages,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
            )
        except Exception as exc:
            logger.exception("generation failed")
            _record_stats("/v1/chat/completions", "500", _empty_stats(), time.perf_counter() - wall_start)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        wall = time.perf_counter() - wall_start
        _record_stats("/v1/chat/completions", "200", stats, wall)
        body = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.completion_tokens,
                "total_tokens": stats.prompt_tokens + stats.completion_tokens,
                "speculative_decoding": {
                    "accepted_tokens": stats.accepted_tokens,
                    "proposed_tokens": stats.proposed_tokens,
                    "acceptance_rate": stats.acceptance_rate,
                },
            },
        }
        return JSONResponse(body)

    # Streaming
    return StreamingResponse(
        _sse_iter(eng, messages, req, request_id, created),
        media_type="text/event-stream",
    )


def _empty_stats() -> GenerationStats:
    return GenerationStats(
        prompt_tokens=0,
        completion_tokens=0,
        ttft_seconds=0.0,
        total_seconds=0.0,
        accepted_tokens=0,
        proposed_tokens=0,
    )


def _sse_iter(
    engine: MTPEngine,
    messages: list[dict[str, str]],
    req: ChatCompletionRequest,
    request_id: str,
    created: int,
) -> Iterator[bytes]:
    wall_start = time.perf_counter()
    final_stats: GenerationStats | None = None
    status = "200"

    def _chunk(delta: dict[str, str], finish: str | None = None, usage: dict | None = None) -> bytes:
        body = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish,
                }
            ],
        }
        if usage is not None:
            body["usage"] = usage
        return f"data: {json.dumps(body)}\n\n".encode()

    try:
        yield _chunk({"role": "assistant"})
        for delta_text, stats in engine.stream_generate(
            messages=messages,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
        ):
            if stats is not None:
                final_stats = stats
                continue
            if delta_text:
                yield _chunk({"content": delta_text})
        usage = None
        if final_stats:
            usage = {
                "prompt_tokens": final_stats.prompt_tokens,
                "completion_tokens": final_stats.completion_tokens,
                "total_tokens": final_stats.prompt_tokens + final_stats.completion_tokens,
                "speculative_decoding": {
                    "accepted_tokens": final_stats.accepted_tokens,
                    "proposed_tokens": final_stats.proposed_tokens,
                    "acceptance_rate": final_stats.acceptance_rate,
                },
            }
        yield _chunk({}, finish="stop", usage=usage)
        yield b"data: [DONE]\n\n"
    except Exception as exc:
        status = "500"
        logger.exception("stream generation failed")
        err = {"error": {"type": "internal_error", "message": str(exc)}}
        yield f"data: {json.dumps(err)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    finally:
        total = time.perf_counter() - wall_start
        _record_stats(
            "/v1/chat/completions",
            status,
            final_stats or _empty_stats(),
            total,
        )
