"""Strict JSON configuration for preparation and experiment runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .clients import OllamaClient, OpenAICompatibleClient, StaticSequenceClient, TextClient
from .probes import stable_hash
from .routing import RoutingPolicy


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a JSON object.")
    return config


def config_hash(config: dict[str, Any]) -> str:
    return stable_hash(config)


def build_client(config: dict[str, Any]) -> TextClient:
    backend = str(config.get("backend", "")).strip()
    if backend == "static":
        responses = config.get("responses")
        if not isinstance(responses, list):
            raise TypeError("static backend requires a responses array.")
        return StaticSequenceClient(
            [str(item) for item in responses],
            model_id=str(config.get("model_id", "static-test")),
        )
    if backend == "ollama":
        return OllamaClient(
            model_id=str(config["model_id"]),
            endpoint=str(config.get("endpoint", "http://localhost:11434/api/chat")),
            timeout_seconds=float(config.get("timeout_seconds", 120.0)),
            num_ctx=int(config.get("num_ctx", 8192)),
            num_predict=int(config.get("num_predict", 100)),
            top_p=float(config.get("top_p", 1.0)),
            top_k=int(config.get("top_k", 0)),
            seed=int(config.get("seed", 42)),
            keep_alive=str(config.get("keep_alive", "5m")),
        )
    if backend == "openai_compatible":
        return OpenAICompatibleClient(
            model_id=str(config["model_id"]),
            base_url=str(config["base_url"]),
            api_key_env=str(config["api_key_env"]),
            timeout_seconds=float(config.get("timeout_seconds", 120.0)),
            max_tokens=int(config.get("max_tokens", 4096)),
            top_p=float(config.get("top_p", 1.0)),
            seed=int(config["seed"]) if config.get("seed") is not None else None,
            thinking_mode=str(config["thinking_mode"]) if config.get("thinking_mode") is not None else None,
            max_retries=int(config.get("max_retries", 4)),
        )
    raise ValueError(f"Unsupported backend: {backend}")


def require_keys(config: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError("Missing required configuration fields: " + ", ".join(missing))


def _validate_model_config(config: dict[str, Any], *, role: str) -> None:
    """Production models are prohibited from relying on hidden default values; `static` is used solely for offline unit testing."""

    require_keys(config, "backend", "model_id")
    backend = str(config["backend"])
    if backend == "static":
        require_keys(config, "responses")
        return
    common = ("temperature", "top_p", "seed", "timeout_seconds")
    if backend == "ollama":
        required = (*common, "endpoint", "top_k", "num_ctx", "num_predict", "keep_alive")
    elif backend == "openai_compatible":
        required = (
            *common,
            "base_url",
            "api_key_env",
            "max_tokens",
            "thinking_mode",
            "max_retries",
        )
    else:
        raise ValueError(f"Unsupported backend for {role}: {backend}")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(
            f"Reviewer reproducibility parameters missing for {role}: {', '.join(missing)}"
        )


def validate_prepare_config(config: dict[str, Any]) -> None:
    require_keys(
        config,
        "dataset",
        "prepared_path",
        "preparation_model",
        "decomposition_format_max_attempts",
        "decomposition_failure_policy",
        "mutation_temperature",
        "mutation_format_max_attempts",
        "mutation_failure_policy",
        "resume",
    )
    dataset = config["dataset"]
    if not isinstance(dataset, dict):
        raise TypeError("dataset must be an object.")
    require_keys(dataset, "path", "format")
    if dataset["format"] not in {"paired_answers", "factoids"}:
        raise ValueError("dataset.format must be paired_answers or factoids.")
    if dataset["format"] == "factoids" and str(dataset["path"]).lower().endswith(".json"):
        
        pass
    preparation_model = config["preparation_model"]
    if not isinstance(preparation_model, dict):
        raise TypeError("preparation_model must be an object.")
    _validate_model_config(preparation_model, role="preparation_model")
    if int(config["decomposition_format_max_attempts"]) < 1:
        raise ValueError("decomposition_format_max_attempts must be positive.")
    if config["decomposition_failure_policy"] not in {
        "fail_fast",
        "record_and_continue",
    }:
        raise ValueError(
            "decomposition_failure_policy must be fail_fast or record_and_continue."
        )
    mutation_temperature = float(config["mutation_temperature"])
    if not 0.0 <= mutation_temperature <= 2.0:
        raise ValueError("mutation_temperature must be in [0, 2].")
    if int(config["mutation_format_max_attempts"]) < 1:
        raise ValueError("mutation_format_max_attempts must be positive.")
    if config["mutation_failure_policy"] not in {"fail_fast", "record_and_continue"}:
        raise ValueError(
            "mutation_failure_policy must be fail_fast or record_and_continue."
        )


def validate_run_config(config: dict[str, Any]) -> None:
    require_keys(
        config,
        "prepared_path",
        "output_root",
        "stage1",
        "routing_policy",
        "scoring_profile",
        "response_aggregation",
    )
    stage1 = config["stage1"]
    if not isinstance(stage1, dict):
        raise TypeError("stage1 must be an object.")
    _validate_model_config(stage1, role="stage1")
    try:
        policy = RoutingPolicy(str(config["routing_policy"]))
    except ValueError as error:
        raise ValueError(f"Unsupported routing_policy: {config['routing_policy']}") from error
    stage2 = config.get("stage2")
    if policy is not RoutingPolicy.NO_ESCALATION and stage2 is None:
        raise ValueError(f"routing_policy={policy.value} requires a Stage-2 verifier.")
    if stage2 is not None:
        if not isinstance(stage2, dict):
            raise TypeError("stage2 must be null or an object.")
        _validate_model_config(stage2, role="stage2")
    if str(stage1["backend"]) != "static":
        environment = config.get("environment")
        if not isinstance(environment, dict):
            raise ValueError("Formal runs require an explicit environment object.")
        require_keys(environment, "concurrency", "execution")
        if int(environment["concurrency"]) != 1:
            raise ValueError("Primary reviewer protocol requires concurrency=1.")
    threshold = float(config.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    if config["scoring_profile"] != "paper_five_statement_mean":
        raise ValueError("scoring_profile must explicitly be paper_five_statement_mean.")
    if config["response_aggregation"] != "max_factoid_score":
        raise ValueError("response_aggregation must explicitly be max_factoid_score.")
