"""DeepSeek-V4-Flash Invoker."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ApiResult:
    ok: bool
    text: str = ""
    error: str = ""
    response_id: str = ""
    returned_model: str = ""
    system_fingerprint: str = ""
    usage: Optional[Dict[str, Any]] = None
    attempts: int = 0
    attempt_errors: List[str] = field(default_factory=list)


class DeepSeekClient:
   

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        thinking_mode: str,
        max_retries: int,
        _sdk_client: Any = None,
    ) -> None:
        self.model = model
        self.thinking_mode = thinking_mode
        self.max_retries = max_retries
        if _sdk_client is not None:
            self.client = _sdk_client
            return

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Please set it in the Environment variables of the PyCharm Run Configuration, do not write it into the code."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing the openai package, please run pip install -r requirements_simple.txt") from exc
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

    def complete(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ApiResult:
        errors = []
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"thinking": {"type": self.thinking_mode}},
                )
                usage = None
                if getattr(response, "usage", None) is not None:
                    usage_object = response.usage
                    usage = (
                        usage_object.model_dump()
                        if hasattr(usage_object, "model_dump")
                        else dict(usage_object)
                    )
                return ApiResult(
                    ok=True,
                    text=(response.choices[0].message.content or "").strip(),
                    response_id=str(getattr(response, "id", "") or ""),
                    returned_model=str(getattr(response, "model", "") or ""),
                    system_fingerprint=str(getattr(response, "system_fingerprint", "") or ""),
                    usage=usage,
                    attempts=attempt,
                    attempt_errors=errors,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                message = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "sk-[REDACTED]", message)
                errors.append(message)
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 30))
        return ApiResult(
            ok=False,
            error=errors[-1] if errors else "unknown_error",
            attempts=self.max_retries,
            attempt_errors=errors,
        )


def parse_boolean(text: str) -> Optional[bool]:
    """Only accept True/False, and avoid misjudging explanatory text as labels."""
    normalized = (text or "").strip().lower()
    normalized = normalized.replace(".", "").replace(",", "").replace(":", "").replace("：", "")
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None

