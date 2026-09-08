"""Context-grounded three-way verifier shared by all experiments."""

from __future__ import annotations

import uuid

from .clients import TextClient
from .cache import VerificationCache
from .labels import parse_verdict
from .prompts import (
    PROMPT_BUNDLE_VERSION,
    VERIFIER_SYSTEM_PROMPT,
    VERIFIER_USER_TEMPLATE,
    prompt_hash,
)
from .probes import stable_hash
from .schema import VerificationObservation


class EvidenceVerifier:
    def __init__(
        self,
        client: TextClient,
        *,
        temperature: float = 0.0,
        cache: VerificationCache | None = None,
    ):
        self.client = client
        self.temperature = temperature
        self.cache = cache

    @property
    def model_id(self) -> str:
        return self.client.model_id

    def verify(
        self,
        *,
        statement: str,
        context: str,
        phase: str,
        statement_type: str,
    ) -> VerificationObservation:
        user_prompt = VERIFIER_USER_TEMPLATE.format(
            context=context.strip(),
            statement=statement.strip(),
        )
        request_payload = {
            "system": VERIFIER_SYSTEM_PROMPT,
            "user": user_prompt,
            "temperature": self.temperature,
            "model_id": self.client.model_id,
        }
        request_hash = stable_hash(request_payload)
        response = self.cache.get(request_hash) if self.cache is not None else None
        cache_hit = response is not None
        if response is None:
            response = self.client.complete(
                system_prompt=VERIFIER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self.temperature,
                phase=phase,
            )
            if self.cache is not None:
                self.cache.put(request_hash, response)
        verdict = parse_verdict(response.content)
        return VerificationObservation(
            call_id=str(uuid.uuid4()),
            model_id=response.model_id,
            backend=response.backend,
            phase=phase,
            statement_type=statement_type,
            statement=statement,
            verdict=verdict,
            raw_response=response.content,
            latency_seconds=response.latency_seconds,
            usage=response.usage,
            request_hash=request_hash,
            response_hash=stable_hash(response.content),
            cache_hit=cache_hit,
            provider_response_id=response.provider_response_id,
            returned_model_id=response.returned_model_id,
            system_fingerprint=response.system_fingerprint,
            finish_reason=response.finish_reason,
            attempts=response.attempts,
            request_parameters=dict(response.request_parameters or {}),
            usage_details=dict(response.usage_details or {}),
            provider_metadata=dict(response.provider_metadata or {}),
            status="success",
            prompt_bundle_version=PROMPT_BUNDLE_VERSION,
            system_prompt_hash=prompt_hash(VERIFIER_SYSTEM_PROMPT),
            user_prompt_hash=prompt_hash(user_prompt),
        )
