"""Gemma 4 MTP engine.

Mirrors the official reference exactly:
  https://huggingface.co/google/gemma-4-E2B-it-assistant
  https://ai.google.dev/gemma/docs/mtp/mtp

The Hugging Face `transformers` library implements the speculative-decoding
loop end-to-end (drafter proposes N tokens autoregressively, target verifies
all N in one forward pass, accept/reject via standard speculative sampling).
We do NOT reimplement that loop here -- we mirror the reference invocation
verbatim and expose it behind an OpenAI-compatible HTTP API.

Per the official Google docs:
  - `assistant_model=assistant_model` is all you need to enable MTP
  - `num_assistant_tokens = 4` (default, per docs)
  - `num_assistant_tokens_schedule = "heuristic"` (default, per docs)
    * all tokens accepted -> +2 tokens to draft next step
    * any tokens rejected -> -1 token to draft next step

We capture, in addition to the reference behavior, the per-request
acceptance statistics that the engine emits so we can report them as a
benchmark metric.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import AsyncIterator, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.generation import candidate_generator as _cg_mod
from transformers.generation.streamers import TextIteratorStreamer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Acceptance counters
# ---------------------------------------------------------------------------
# transformers does not expose accept/propose totals after a generate() call.
# We monkey-patch AssistedCandidateGenerator.update_candidate_strategy so that
# every call accumulates into a thread-local counter that the engine reads.
#
# update_candidate_strategy is invoked once per spec-decode iteration with
# `num_matches` = number of accepted draft tokens this step, and
# `len(scores[0]) - 1` = number of proposed draft tokens this step.
#
# This is intentionally surgical: we keep the official reference loop and
# only observe its counters.
# ---------------------------------------------------------------------------


class _SpecCounters:
    __slots__ = ("accepted", "proposed")

    def __init__(self) -> None:
        self.accepted = 0
        self.proposed = 0

    def reset(self) -> None:
        self.accepted = 0
        self.proposed = 0


# NOTE: previously thread-local; uvicorn dispatches generate() inside a
# worker thread distinct from the request handler thread, which made the
# counters invisible to the caller. The engine already serializes generate()
# behind a lock, so a single global counter is correct and simpler.
_global_counters = _SpecCounters()


def _get_counters() -> _SpecCounters:
    return _global_counters


_orig_update = _cg_mod.AssistedCandidateGenerator.update_candidate_strategy


def _wrapped_update(self, input_ids, scores, num_matches):  # type: ignore[no-redef]
    try:
        proposed = int(scores.shape[1]) - 1 if hasattr(scores, "shape") else max(0, len(scores[0]) - 1)
    except Exception:
        proposed = 0
    c = _get_counters()
    c.proposed += max(0, proposed)
    c.accepted += max(0, int(num_matches))
    return _orig_update(self, input_ids, scores, num_matches)


_cg_mod.AssistedCandidateGenerator.update_candidate_strategy = _wrapped_update


# Exactly the IDs from the Gemma 4 MTP reference code.
TARGET_MODEL_ID = os.getenv("TARGET_MODEL", "google/gemma-4-E2B-it")
ASSISTANT_MODEL_ID = os.getenv("ASSISTANT_MODEL", "google/gemma-4-E2B-it-assistant")

# Per ai.google.dev/gemma/docs/mtp/mtp -- defaults specified in the docs.
NUM_ASSISTANT_TOKENS = int(os.getenv("NUM_ASSISTANT_TOKENS", "4"))
NUM_ASSISTANT_TOKENS_SCHEDULE = os.getenv("NUM_ASSISTANT_TOKENS_SCHEDULE", "heuristic")


@dataclass
class GenerationStats:
    prompt_tokens: int
    completion_tokens: int
    ttft_seconds: float
    total_seconds: float
    accepted_tokens: int
    proposed_tokens: int

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted_tokens / self.proposed_tokens) if self.proposed_tokens else 0.0


class MTPEngine:
    """Wraps the Gemma 4 target + assistant pair behind a single class.

    Loads both models once at startup (per the reference snippet) and
    serves generate() and stream_generate() calls.
    """

    def __init__(
        self,
        target_model_id: str = TARGET_MODEL_ID,
        assistant_model_id: str = ASSISTANT_MODEL_ID,
        num_assistant_tokens: int = NUM_ASSISTANT_TOKENS,
        num_assistant_tokens_schedule: str = NUM_ASSISTANT_TOKENS_SCHEDULE,
    ) -> None:
        logger.info("loading target model %s", target_model_id)
        self.processor = AutoProcessor.from_pretrained(target_model_id)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            dtype="auto",
            device_map="auto",
        )
        logger.info("loading assistant (drafter) model %s", assistant_model_id)
        self.assistant_model = AutoModelForCausalLM.from_pretrained(
            assistant_model_id,
            dtype="auto",
            device_map="auto",
        )
        # Per Google docs: tune draft budget on the assistant model's config.
        self.assistant_model.generation_config.num_assistant_tokens = num_assistant_tokens
        self.assistant_model.generation_config.num_assistant_tokens_schedule = (
            num_assistant_tokens_schedule
        )
        # Lock for serial-only inference; the target model is not thread-safe.
        self._lock = threading.Lock()
        self.target_model_id = target_model_id
        self.assistant_model_id = assistant_model_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_inputs(self, messages: list[dict[str, str]]) -> tuple[dict[str, torch.Tensor], int]:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.target_model.device)
        input_len = int(inputs["input_ids"].shape[-1])
        return inputs, input_len

    def _generation_kwargs(
        self,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> dict[str, object]:
        do_sample = temperature is not None and temperature > 0.0
        kwargs: dict[str, object] = {
            "assistant_model": self.assistant_model,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "return_dict_in_generate": True,
        }
        if do_sample:
            kwargs["temperature"] = float(temperature)
            kwargs["top_p"] = float(top_p)
            kwargs["top_k"] = int(top_k)
        return kwargs

    @staticmethod
    def _extract_acceptance(_target_model: torch.nn.Module) -> tuple[int, int]:
        """Read accept/propose totals from the thread-local _SpecCounters."""
        c = _get_counters()
        return c.accepted, c.proposed

    # ------------------------------------------------------------------
    # Non-streaming generate
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> tuple[str, GenerationStats]:
        with self._lock:
            inputs, input_len = self._build_inputs(messages)
            gen_kwargs = self._generation_kwargs(max_new_tokens, temperature, top_p, top_k)

            # Reset counters before this call.
            self._reset_counters()

            t0 = time.perf_counter()
            outputs = self.target_model.generate(**inputs, **gen_kwargs)
            t1 = time.perf_counter()

            sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
            response_ids = sequences[0][input_len:]
            response = self.processor.decode(response_ids, skip_special_tokens=False)

            accepted, proposed = self._extract_acceptance(self.target_model)
            stats = GenerationStats(
                prompt_tokens=input_len,
                completion_tokens=int(response_ids.shape[-1]),
                ttft_seconds=t1 - t0,  # No streaming -> TTFT == E2E for non-stream.
                total_seconds=t1 - t0,
                accepted_tokens=accepted,
                proposed_tokens=proposed,
            )
            return response, stats

    # ------------------------------------------------------------------
    # Streaming generate
    # ------------------------------------------------------------------

    def stream_generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> Iterator[tuple[str, GenerationStats | None]]:
        """Yield (delta_text, stats_or_none).

        Final element has stats != None and an empty delta.
        """
        with self._lock:
            inputs, input_len = self._build_inputs(messages)
            gen_kwargs = self._generation_kwargs(max_new_tokens, temperature, top_p, top_k)

            streamer = TextIteratorStreamer(
                self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor,
                skip_prompt=True,
                skip_special_tokens=False,
            )
            gen_kwargs["streamer"] = streamer

            self._reset_counters()
            start = time.perf_counter()
            ttft: float | None = None
            completion_tokens = 0

            thread = threading.Thread(
                target=self.target_model.generate,
                kwargs={**inputs, **gen_kwargs},
            )
            thread.start()

            for delta in streamer:
                if not delta:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - start
                completion_tokens += 1  # Approximate: streamer yields per-step.
                yield delta, None

            thread.join()
            end = time.perf_counter()
            accepted, proposed = self._extract_acceptance(self.target_model)
            stats = GenerationStats(
                prompt_tokens=input_len,
                completion_tokens=completion_tokens,
                ttft_seconds=ttft or 0.0,
                total_seconds=end - start,
                accepted_tokens=accepted,
                proposed_tokens=proposed,
            )
            yield "", stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset_counters(self) -> None:
        _get_counters().reset()


# Module-level lazy singleton so FastAPI workers reuse one engine.
_engine_lock = threading.Lock()
_engine: MTPEngine | None = None


def get_engine() -> MTPEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = MTPEngine()
    return _engine
