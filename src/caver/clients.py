"""Backend clients with explicit token, call, and latency accounting."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .schema import CallUsage


@dataclass(frozen=True)
class TextResponse:
    content: str
    usage: CallUsage
    latency_seconds: float
    backend: str
    model_id: str
    provider_response_id: str = ""
    returned_model_id: str = ""
    system_fingerprint: str = ""
    finish_reason: str = ""
    attempts: int = 1
    request_parameters: Mapping[str, Any] | None = None
    usage_details: Mapping[str, Any] | None = None
    provider_metadata: Mapping[str, Any] | None = None


class TextClient(Protocol):
    backend: str
    model_id: str

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        phase: str,
    ) -> TextResponse:
        ...


class OpenAICompatibleClient:
    backend = "openai_compatible"

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str,
        api_key_env: str,
        timeout_seconds: float = 120.0,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        seed: int | None = None,
        thinking_mode: str | None = None,
        max_retries: int = 4,
    ):
        model_id = model_id.strip()
        if not model_id or model_id.startswith("REPLACE_"):
            raise ValueError("An exact API model identifier is required.")
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"Required API key environment variable is empty: {api_key_env}")
        from openai import OpenAI

        self.model_id = model_id
        self.base_url = str(base_url)
        self.timeout_seconds = float(timeout_seconds)
        self.max_tokens = int(max_tokens)
        self.top_p = float(top_p)
        self.seed = seed
        self.thinking_mode = thinking_mode
        self.max_retries = int(max_retries)
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        if self.max_retries < 1:
            raise ValueError("max_retries must be positive.")
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        phase: str,
    ) -> TextResponse:
        parameters: dict[str, Any] = {
            "model": self.model_id,
            "temperature": float(temperature),
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "thinking_mode": self.thinking_mode,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "phase": phase,
        }
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        if self.thinking_mode is not None:
            kwargs["extra_body"] = {"thinking": {"type": self.thinking_mode}}

        response = None
        last_error: Exception | None = None
        elapsed = 0.0
        attempts = 0
        for attempts in range(1, self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._client.chat.completions.create(**kwargs)
                elapsed += time.perf_counter() - started
                break
            except Exception as error: 
                elapsed += time.perf_counter() - started
                last_error = error
                if attempts < self.max_retries:
                    time.sleep(min(2 ** (attempts - 1), 8))
        if response is None:
            assert last_error is not None
            raise RuntimeError(
                f"Remote completion failed after {attempts} attempts: "
                f"{type(last_error).__name__}: {last_error}"
            ) from last_error
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        content = response.choices[0].message.content or ""
        usage_details: dict[str, Any] = {}
        if usage is not None:
            if hasattr(usage, "model_dump"):
                usage_details = usage.model_dump()
            elif isinstance(usage, Mapping):
                usage_details = dict(usage)
        choice = response.choices[0]
        return TextResponse(
            content=content.strip(),
            usage=CallUsage(input_tokens=input_tokens, output_tokens=output_tokens, calls=1),
            latency_seconds=elapsed,
            backend=self.backend,
            model_id=self.model_id,
            provider_response_id=str(getattr(response, "id", "") or ""),
            returned_model_id=str(getattr(response, "model", "") or ""),
            system_fingerprint=str(getattr(response, "system_fingerprint", "") or ""),
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            attempts=attempts,
            request_parameters=parameters,
            usage_details=usage_details,
        )


class OllamaClient:
    backend = "ollama"

    def __init__(
        self,
        *,
        model_id: str,
        endpoint: str = "http://localhost:11434/api/chat",
        timeout_seconds: float = 120.0,
        num_ctx: int = 8192,
        num_predict: int = 100,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int = 42,
        keep_alive: str = "5m",
    ):
        model_id = model_id.strip()
        if not model_id or model_id.startswith("REPLACE_"):
            raise ValueError("An exact Ollama model identifier is required.")
        self.model_id = model_id
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.seed = int(seed)
        self.keep_alive = str(keep_alive)

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        phase: str,
    ) -> TextResponse:
        import requests

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "seed": self.seed,
            },
        }
        started = time.perf_counter()
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout_seconds)
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        data = response.json()
        return TextResponse(
            content=str(data.get("message", {}).get("content", "")).strip(),
            usage=CallUsage(
                input_tokens=int(data.get("prompt_eval_count", 0) or 0),
                output_tokens=int(data.get("eval_count", 0) or 0),
                calls=1,
            ),
            latency_seconds=elapsed,
            backend=self.backend,
            model_id=self.model_id,
            returned_model_id=str(data.get("model", self.model_id)),
            finish_reason=str(data.get("done_reason", "")),
            request_parameters={
                "model": self.model_id,
                "temperature": float(temperature),
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "seed": self.seed,
                "keep_alive": self.keep_alive,
                "endpoint": self.endpoint,
                "timeout_seconds": self.timeout_seconds,
                "phase": phase,
            },
            usage_details={
                "prompt_eval_count": int(data.get("prompt_eval_count", 0) or 0),
                "eval_count": int(data.get("eval_count", 0) or 0),
            },
            provider_metadata={
                "created_at": data.get("created_at"),
                "total_duration_ns": int(data.get("total_duration", 0) or 0),
                "load_duration_ns": int(data.get("load_duration", 0) or 0),
                "prompt_eval_duration_ns": int(data.get("prompt_eval_duration", 0) or 0),
                "eval_duration_ns": int(data.get("eval_duration", 0) or 0),
            },
        )


class StaticSequenceClient:
    """Offline test client that returns a predefined sequence."""

    backend = "static"

    def __init__(self, responses: list[str], model_id: str = "static-test"):
        self.model_id = model_id
        self._responses = list(responses)
        self._index = 0

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        phase: str,
    ) -> TextResponse:
        del system_prompt, user_prompt
        if self._index >= len(self._responses):
            raise RuntimeError("StaticSequenceClient has no remaining responses.")
        content = self._responses[self._index]
        self._index += 1
        return TextResponse(
            content=content,
            usage=CallUsage(input_tokens=1, output_tokens=1, calls=1),
            latency_seconds=0.0,
            backend=self.backend,
            model_id=self.model_id,
            returned_model_id=self.model_id,
            request_parameters={"temperature": float(temperature), "phase": phase},
        )
