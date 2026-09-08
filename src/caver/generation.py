"""Factoid decomposition and mutation generation for the preparation stage."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from .clients import TextClient, TextResponse
from .prompts import (
    ANTONYM_SYSTEM_PROMPT,
    DECOMPOSITION_SYSTEM_PROMPT,
    DECOMPOSITION_USER_TEMPLATE,
    MUTATION_USER_TEMPLATE,
    SYNONYM_SYSTEM_PROMPT,
)
from .probes import make_probe_set
from .schema import ProbeSet


@dataclass(frozen=True)
class DecompositionAttempt:

    system_prompt: str
    user_prompt: str
    generation_attempt: int
    response: TextResponse
    format_valid: bool
    validation_error: str | None


class DecompositionFormatError(ValueError):
    """The API call succeeded, but repeated responses were still not arrays of non-empty JSON strings."""

    def __init__(self, *, attempts: int, validation_error: str) -> None:
        self.attempts = int(attempts)
        self.validation_error = validation_error
        super().__init__(
            "factoid decomposition failed JSON-array validation after "
            f"{attempts} generation attempts: {validation_error}"
        )


@dataclass(frozen=True)
class MutationAttempt:
    """An attempt to generate format-compliant, auditable mutations."""

    phase: str
    system_prompt: str
    user_prompt: str
    generation_attempt: int
    response: TextResponse
    format_valid: bool
    validation_error: str | None


class MutationFormatError(ValueError):
    """The API call succeeded, but repeated responses still failed to match the two fixed variant formats."""

    def __init__(
        self,
        *,
        factoid_id: str,
        phase: str,
        attempts: int,
        validation_error: str,
    ) -> None:
        self.factoid_id = factoid_id
        self.phase = phase
        self.attempts = int(attempts)
        self.validation_error = validation_error
        super().__init__(
            f"{phase} for {factoid_id} failed the exact-two format after "
            f"{attempts} generation attempts: {validation_error}"
        )


def clean_json_array(text: str) -> list[str]:
    fence = chr(96) * 3
    cleaned = text.strip().replace(fence + "json", "").replace(fence, "").strip()
    data = json.loads(cleaned)
    if not isinstance(data, list) or not data:
        raise ValueError("Decomposer must return a non-empty JSON array.")
    values = [str(item).strip() for item in data if str(item).strip()]
    if not values:
        raise ValueError("Decomposer returned no non-empty factoids.")
    return values


def clean_mutation_lines(text: str, expected: int = 2) -> tuple[str, str]:
    values: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if cleaned:
            values.append(cleaned)
    if len(values) != expected:
        raise ValueError(f"Expected exactly {expected} mutations, received {len(values)}.")
    return values[0], values[1]


class FactoidDecomposer:
    def __init__(self, client: TextClient, *, format_max_attempts: int = 3):
        self.client = client
        self.format_max_attempts = int(format_max_attempts)
        if self.format_max_attempts < 1:
            raise ValueError("format_max_attempts must be positive.")

    def decompose(
        self,
        answer: str,
        *,
        on_attempt: Callable[[DecompositionAttempt], None] | None = None,
    ) -> list[str]:
        user_prompt = DECOMPOSITION_USER_TEMPLATE.format(answer=answer.strip())
        last_error: ValueError | json.JSONDecodeError | None = None
        for generation_attempt in range(1, self.format_max_attempts + 1):
            response = self.client.complete(
                system_prompt=DECOMPOSITION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
                phase="decompose",
            )
            try:
                factoids = clean_json_array(response.content)
            except (ValueError, json.JSONDecodeError) as error:
                last_error = error
                attempt = DecompositionAttempt(
                    system_prompt=DECOMPOSITION_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    generation_attempt=generation_attempt,
                    response=response,
                    format_valid=False,
                    validation_error=f"{type(error).__name__}: {error}",
                )
                if on_attempt is not None:
                    on_attempt(attempt)
                continue

            attempt = DecompositionAttempt(
                system_prompt=DECOMPOSITION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                generation_attempt=generation_attempt,
                response=response,
                format_valid=True,
                validation_error=None,
            )
            if on_attempt is not None:
                on_attempt(attempt)
            return factoids

        assert last_error is not None
        raise DecompositionFormatError(
            attempts=self.format_max_attempts,
            validation_error=f"{type(last_error).__name__}: {last_error}",
        ) from last_error


class MutationGenerator:
    def __init__(
        self,
        client: TextClient,
        *,
        temperature: float = 0.2,
        format_max_attempts: int = 3,
    ):
        self.client = client
        self.temperature = temperature
        self.format_max_attempts = int(format_max_attempts)
        if self.format_max_attempts < 1:
            raise ValueError("format_max_attempts must be positive.")

    def _generate_two(
        self,
        *,
        factoid_id: str,
        phase: str,
        system_prompt: str,
        user_prompt: str,
        on_attempt: Callable[[MutationAttempt], None] | None,
    ) -> tuple[str, str]:
        last_error: ValueError | None = None
        for generation_attempt in range(1, self.format_max_attempts + 1):
            response = self.client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                phase=phase,
            )
            try:
                values = clean_mutation_lines(response.content)
            except ValueError as error:
                last_error = error
                attempt = MutationAttempt(
                    phase=phase,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    generation_attempt=generation_attempt,
                    response=response,
                    format_valid=False,
                    validation_error=str(error),
                )
                if on_attempt is not None:
                    on_attempt(attempt)
                continue

            attempt = MutationAttempt(
                phase=phase,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                generation_attempt=generation_attempt,
                response=response,
                format_valid=True,
                validation_error=None,
            )
            if on_attempt is not None:
                on_attempt(attempt)
            return values

        assert last_error is not None
        raise MutationFormatError(
            factoid_id=factoid_id,
            phase=phase,
            attempts=self.format_max_attempts,
            validation_error=str(last_error),
        ) from last_error

    def generate(
        self,
        *,
        factoid_id: str,
        question: str,
        factoid: str,
        on_attempt: Callable[[MutationAttempt], None] | None = None,
    ) -> ProbeSet:
        user_prompt = MUTATION_USER_TEMPLATE.format(
            question=question.strip(),
            factoid=factoid.strip(),
        )
        synonyms = self._generate_two(
            factoid_id=factoid_id,
            phase="mutation_synonym",
            system_prompt=SYNONYM_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            on_attempt=on_attempt,
        )
        antonyms = self._generate_two(
            factoid_id=factoid_id,
            phase="mutation_antonym",
            system_prompt=ANTONYM_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            on_attempt=on_attempt,
        )
        return make_probe_set(
            factoid_id,
            factoid,
            synonyms,
            antonyms,
        )
