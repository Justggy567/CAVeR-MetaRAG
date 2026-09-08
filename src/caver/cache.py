"""Persistent verifier response cache for matched offline policy evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .clients import TextResponse
from .schema import CallUsage


class VerificationCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._items: dict[str, TextResponse] = {}

    def load(self) -> "VerificationCache":
        self._items.clear()
        if not self.path.exists():
            return self
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                data: dict[str, Any] = json.loads(line)
                key = str(data["key"])
                if key in self._items:
                    raise ValueError(f"Duplicate verification cache key at line {line_number}.")
                self._items[key] = TextResponse(
                    content=str(data["content"]),
                    usage=CallUsage(
                        input_tokens=int(data["input_tokens"]),
                        output_tokens=int(data["output_tokens"]),
                        calls=int(data.get("calls", 1)),
                        charge=float(data["charge"]) if data.get("charge") is not None else None,
                    ),
                    latency_seconds=float(data["latency_seconds"]),
                    backend=str(data["backend"]),
                    model_id=str(data["model_id"]),
                    provider_response_id=str(data.get("provider_response_id", "")),
                    returned_model_id=str(data.get("returned_model_id", "")),
                    system_fingerprint=str(data.get("system_fingerprint", "")),
                    finish_reason=str(data.get("finish_reason", "")),
                    attempts=int(data.get("attempts", 1)),
                    request_parameters=dict(data.get("request_parameters", {})),
                    usage_details=dict(data.get("usage_details", {})),
                    provider_metadata=dict(data.get("provider_metadata", {})),
                )
        return self

    def get(self, key: str) -> TextResponse | None:
        return self._items.get(key)

    def put(self, key: str, response: TextResponse) -> None:
        existing = self._items.get(key)
        if existing is not None:
            if existing.content != response.content:
                raise ValueError("Refusing to overwrite a verifier cache key with a different response.")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "key": key,
            "content": response.content,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "calls": response.usage.calls,
            "charge": response.usage.charge,
            "latency_seconds": response.latency_seconds,
            "backend": response.backend,
            "model_id": response.model_id,
            "provider_response_id": response.provider_response_id,
            "returned_model_id": response.returned_model_id,
            "system_fingerprint": response.system_fingerprint,
            "finish_reason": response.finish_reason,
            "attempts": response.attempts,
            "request_parameters": dict(response.request_parameters or {}),
            "usage_details": dict(response.usage_details or {}),
            "provider_metadata": dict(response.provider_metadata or {}),
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
        self._items[key] = response

    def __len__(self) -> int:
        return len(self._items)
