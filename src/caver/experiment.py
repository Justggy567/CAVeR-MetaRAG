"""Preparation and execution functions used by all RQ entry points."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cascade import CanonicalCascade
from .cache import VerificationCache
from .configuration import (
    build_client,
    config_hash,
    validate_prepare_config,
    validate_run_config,
)
from .datasets import (
    PreparedResponse,
    file_sha256,
    load_prepared_responses,
    load_source_responses,
)
from .generation import (
    DecompositionAttempt,
    DecompositionFormatError,
    FactoidDecomposer,
    MutationAttempt,
    MutationFormatError,
    MutationGenerator,
)
from .ledger import RunLedger, atomic_write_json
from .metrics import summarize_serialized
from .probes import probe_warnings, stable_hash
from .prompts import prompt_manifest
from .prompts import (
    PROMPT_BUNDLE_VERSION,
    VERIFIER_SYSTEM_PROMPT,
    VERIFIER_USER_TEMPLATE,
    prompt_hash,
)
from .routing import RoutingPolicy
from .verifiers import EvidenceVerifier
from .taxonomy import taxonomy_manifest


def _git_commit(repository_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()


def _response_audit_fields(response: Any) -> dict[str, Any]:
    """Record a model invocation in a way that is auditable and free of API keys."""

    return {
        "status": "success",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": response.backend,
        "requested_model_id": response.model_id,
        "returned_model_id": response.returned_model_id,
        "provider_response_id": response.provider_response_id,
        "system_fingerprint": response.system_fingerprint,
        "finish_reason": response.finish_reason,
        "attempts": response.attempts,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
        "calls": response.usage.calls,
        "latency_seconds": response.latency_seconds,
        "request_parameters": dict(response.request_parameters or {}),
        "usage_details": dict(response.usage_details or {}),
        "provider_metadata": dict(response.provider_metadata or {}),
        "response_hash": stable_hash(response.content),
    }


def _prompt_audit_fields(system_prompt: str, user_prompt: str) -> dict[str, str]:
    return {
        "prompt_bundle_version": PROMPT_BUNDLE_VERSION,
        "system_prompt_hash": prompt_hash(system_prompt),
        "user_prompt_hash": prompt_hash(user_prompt),
        "request_hash": stable_hash(
            {"system": system_prompt, "user": user_prompt}
        ),
    }


def _reproducibility_protocol(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "configs" / "reproducibility_protocol.json"
    if not path.exists():
        path = Path(__file__).resolve().parents[2] / "configs" / "reproducibility_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise TypeError("reproducibility_protocol.json must contain a JSON object.")
    return protocol, file_sha256(path)


def prepare_from_config(config: dict[str, Any], *, repository_root: str | Path) -> Path:
    validate_prepare_config(config)
    root = Path(repository_root)
    dataset_config = config["dataset"]
    dataset_path = root / str(dataset_config["path"])
    prepared_path = root / str(config["prepared_path"])
    if prepared_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed prepared data: {prepared_path}")
    partial_path = prepared_path.with_suffix(prepared_path.suffix + ".partial")
    call_path = prepared_path.with_suffix(prepared_path.suffix + ".calls.jsonl")
    unresolved_path = prepared_path.with_suffix(
        prepared_path.suffix + ".unresolved_mutations.jsonl"
    )
    unresolved_decomposition_path = prepared_path.with_suffix(
        prepared_path.suffix + ".unresolved_decompositions.jsonl"
    )
    manifest_path = prepared_path.with_suffix(prepared_path.suffix + ".manifest.json")
    resume = bool(config.get("resume", False))
    if (
        partial_path.exists()
        or call_path.exists()
        or unresolved_path.exists()
        or unresolved_decomposition_path.exists()
        or manifest_path.exists()
    ) and not resume:
        raise FileExistsError(
            "Preparation artifacts already exist. Set resume=true only for the same frozen config."
        )
    source_responses = load_source_responses(
        dataset_path,
        dataset_format=str(dataset_config["format"]),
        answer_roles=str(dataset_config.get("answer_roles", "both")),
        dataset_name=str(dataset_config.get("name", Path(dataset_path).stem)),
        limit=int(dataset_config["limit"]) if dataset_config.get("limit") is not None else None,
    )
    client = build_client(config["preparation_model"])
    decomposer = FactoidDecomposer(
        client,
        format_max_attempts=int(config["decomposition_format_max_attempts"]),
    )
    generator = MutationGenerator(
        client,
        temperature=float(config["mutation_temperature"]),
        format_max_attempts=int(config["mutation_format_max_attempts"]),
    )
    prepared: list[PreparedResponse] = (
        load_prepared_responses(partial_path) if partial_path.exists() and partial_path.stat().st_size else []
    )
    unresolved_rows = _read_jsonl(unresolved_path)
    unresolved_by_factoid: dict[str, dict[str, Any]] = {}
    for row in unresolved_rows:
        factoid_id = str(row.get("factoid_id", ""))
        if not factoid_id:
            raise ValueError("Unresolved mutation checkpoint contains an empty factoid_id.")
        if factoid_id in unresolved_by_factoid:
            raise ValueError(
                f"Unresolved mutation checkpoint contains duplicate factoid_id: {factoid_id}"
            )
        unresolved_by_factoid[factoid_id] = row
    unresolved_decomposition_rows = _read_jsonl(unresolved_decomposition_path)
    unresolved_decomposition_by_response: dict[str, dict[str, Any]] = {}
    for row in unresolved_decomposition_rows:
        response_id = str(row.get("response_id", ""))
        if not response_id:
            raise ValueError(
                "Unresolved decomposition checkpoint contains an empty response_id."
            )
        if response_id in unresolved_decomposition_by_response:
            raise ValueError(
                "Unresolved decomposition checkpoint contains duplicate response_id: "
                f"{response_id}"
            )
        unresolved_decomposition_by_response[response_id] = row
    completed_ids = {item.response_id for item in prepared}
    source_ids = {item.response_id for item in source_responses}
    if not completed_ids.issubset(source_ids):
        raise ValueError("Preparation checkpoint contains response IDs absent from the source dataset.")
    dataset_digest = file_sha256(dataset_path)
    protocol, protocol_sha256 = _reproducibility_protocol(root)
    current_config_hash = config_hash(config)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "config_hash": current_config_hash,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_digest,
        "prepared_path": str(prepared_path),
        "source_records": len(source_responses),
        "preparation_model": config["preparation_model"],
        "decomposition_format_max_attempts": int(
            config["decomposition_format_max_attempts"]
        ),
        "decomposition_failure_policy": str(config["decomposition_failure_policy"]),
        "mutation_temperature": float(config["mutation_temperature"]),
        "mutation_format_max_attempts": int(config["mutation_format_max_attempts"]),
        "mutation_failure_policy": str(config["mutation_failure_policy"]),
        "mutation_format_policy": (
            "exactly two non-empty lines after bullet/number-prefix stripping; "
            "retry the same frozen prompt on format failure; after exhaustion, "
            "record the unresolved factoid and continue only when configured"
        ),
        "prompt_bundle": prompt_manifest(),
        "rq3_taxonomy": taxonomy_manifest(),
        "reproducibility_protocol": protocol,
        "reproducibility_protocol_sha256": protocol_sha256,
        "status": "running",
        "completed_responses": len(prepared),
        "unresolved_mutations": len(unresolved_by_factoid),
        "unresolved_mutations_path": str(unresolved_path),
        "unresolved_decompositions": len(unresolved_decomposition_by_response),
        "unresolved_decompositions_path": str(unresolved_decomposition_path),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "config_hash",
            "dataset_sha256",
            "reproducibility_protocol_sha256",
        ):
            if existing.get(key) != manifest.get(key):
                raise ValueError(f"Preparation resume manifest mismatch for {key}.")
        manifest = existing
        manifest["status"] = "running"
        manifest["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(manifest_path, manifest)

    try:
        for source in source_responses:
            if source.response_id in completed_ids:
                continue
            if source.frozen_probes:
                probe_sets = source.frozen_probes
                preparation_metadata = {
                    "source_factoid_count": len(probe_sets),
                    "prepared_factoid_count": len(probe_sets),
                    "unresolved_mutation_count": 0,
                    "unresolved_factoid_ids": [],
                    "decomposition_unresolved": False,
                    "analysis_eligible": bool(probe_sets),
                }
            else:
                decomposition_unresolved = False
                if source.source_factoids:
                    factoids = list(source.source_factoids)
                else:
                    checkpoint = unresolved_decomposition_by_response.get(
                        source.response_id
                    )
                    if checkpoint is not None:
                        if str(checkpoint.get("answer", "")) != source.answer:
                            raise ValueError(
                                "Source answer changed across resume for unresolved "
                                f"decomposition {source.response_id}."
                            )
                        factoids = []
                        decomposition_unresolved = True
                    else:

                        def record_decomposition_attempt(
                            attempt: DecompositionAttempt,
                        ) -> None:
                            _append_jsonl(
                                call_path,
                                {
                                    "phase": "decompose",
                                    "response_id": source.response_id,
                                    "generation_attempt": attempt.generation_attempt,
                                    "format_valid": attempt.format_valid,
                                    "validation_error": attempt.validation_error,
                                    "response_content": attempt.response.content,
                                    **_prompt_audit_fields(
                                        attempt.system_prompt,
                                        attempt.user_prompt,
                                    ),
                                    **_response_audit_fields(attempt.response),
                                },
                            )

                        try:
                            factoids = decomposer.decompose(
                                source.answer,
                                on_attempt=record_decomposition_attempt,
                            )
                        except DecompositionFormatError as error:
                            if config["decomposition_failure_policy"] == "fail_fast":
                                raise
                            unresolved_decomposition = {
                                "record_type": "unresolved_decomposition",
                                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                                "response_id": source.response_id,
                                "question_id": source.question_id,
                                "answer_pair_id": source.answer_pair_id,
                                "answer_role": source.answer_role,
                                "answer": source.answer,
                                "answer_character_count": len(source.answer),
                                "generation_attempts": error.attempts,
                                "validation_error": error.validation_error,
                                "error_type": type(error).__name__,
                                "disposition": "response_excluded_no_valid_factoids",
                            }
                            _append_jsonl(
                                unresolved_decomposition_path,
                                unresolved_decomposition,
                            )
                            unresolved_decomposition_by_response[source.response_id] = (
                                unresolved_decomposition
                            )
                            manifest["unresolved_decompositions"] = len(
                                unresolved_decomposition_by_response
                            )
                            atomic_write_json(manifest_path, manifest)
                            factoids = []
                            decomposition_unresolved = True
                generated = []
                response_unresolved_ids: list[str] = []
                for index, factoid in enumerate(factoids):
                    factoid_id = source.response_id + ":f" + str(index)

                    # If the previous run exhausted all format attempts, resuming from the breakpoint will not trigger repeated paid calls.
                    if factoid_id in unresolved_by_factoid:
                        checkpoint_factoid = str(
                            unresolved_by_factoid[factoid_id].get("factoid", "")
                        )
                        if checkpoint_factoid != factoid:
                            raise ValueError(
                                "Factoid decomposition changed across resume for "
                                f"{factoid_id}; refusing to apply an unresolved record "
                                "to different text."
                            )
                        response_unresolved_ids.append(factoid_id)
                        continue

                    def record_mutation_attempt(attempt: MutationAttempt) -> None:
                        # Record the transaction immediately after every successful API response, including responses with invalid formats.
                        _append_jsonl(
                            call_path,
                            {
                                "phase": attempt.phase,
                                "response_id": source.response_id,
                                "factoid_id": factoid_id,
                                "generation_attempt": attempt.generation_attempt,
                                "format_valid": attempt.format_valid,
                                "validation_error": attempt.validation_error,
                                "response_content": attempt.response.content,
                                **_prompt_audit_fields(
                                    attempt.system_prompt,
                                    attempt.user_prompt,
                                ),
                                **_response_audit_fields(attempt.response),
                            },
                        )

                    try:
                        probes = generator.generate(
                            factoid_id=factoid_id,
                            question=source.question,
                            factoid=factoid,
                            on_attempt=record_mutation_attempt,
                        )
                    except MutationFormatError as error:
                        if config["mutation_failure_policy"] == "fail_fast":
                            raise
                        unresolved = {
                            "record_type": "unresolved_mutation",
                            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                            "response_id": source.response_id,
                            "question_id": source.question_id,
                            "answer_pair_id": source.answer_pair_id,
                            "answer_role": source.answer_role,
                            "factoid_id": factoid_id,
                            "factoid_index": index,
                            "factoid": factoid,
                            "failure_phase": error.phase,
                            "generation_attempts": error.attempts,
                            "validation_error": error.validation_error,
                            "error_type": type(error).__name__,
                            "disposition": "excluded_from_prepared_probes",
                        }
                        _append_jsonl(unresolved_path, unresolved)
                        unresolved_by_factoid[factoid_id] = unresolved
                        response_unresolved_ids.append(factoid_id)
                        manifest["unresolved_mutations"] = len(unresolved_by_factoid)
                        atomic_write_json(manifest_path, manifest)
                        continue
                    generated.append(probes)
                probe_sets = tuple(generated)
                preparation_metadata = {
                    "source_factoid_count": len(factoids),
                    "prepared_factoid_count": len(probe_sets),
                    "unresolved_mutation_count": len(response_unresolved_ids),
                    "unresolved_factoid_ids": response_unresolved_ids,
                    "decomposition_unresolved": decomposition_unresolved,
                    "analysis_eligible": bool(probe_sets),
                }
            response_metadata = dict(source.metadata or {})
            response_metadata["preparation"] = preparation_metadata
            prepared_response = PreparedResponse(
                response_id=source.response_id,
                question_id=source.question_id,
                answer_pair_id=source.answer_pair_id,
                answer_role=source.answer_role,
                question=source.question,
                context=source.context,
                answer=source.answer,
                gold_target=source.gold_target,
                probe_sets=probe_sets,
                metadata=response_metadata,
            )
            _append_jsonl(partial_path, prepared_response.to_dict())
            prepared.append(prepared_response)
            completed_ids.add(source.response_id)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["completed_responses"] = len(prepared)
        manifest["unresolved_mutations"] = len(unresolved_by_factoid)
        manifest["unresolved_decompositions"] = len(
            unresolved_decomposition_by_response
        )
        manifest["last_error"] = f"{type(error).__name__}: {error}"
        atomic_write_json(manifest_path, manifest)
        raise

    if len(prepared) != len(source_responses):
        raise AssertionError("Prepared checkpoint count does not match normalized source responses.")
    referenced_unresolved_ids = {
        str(factoid_id)
        for item in prepared
        for factoid_id in (
            (item.metadata.get("preparation") or {}).get("unresolved_factoid_ids", [])
        )
    }
    if referenced_unresolved_ids != set(unresolved_by_factoid):
        raise AssertionError(
            "Prepared response metadata and unresolved mutation ledger do not match."
        )
    referenced_unresolved_decompositions = {
        item.response_id
        for item in prepared
        if bool(
            (item.metadata.get("preparation") or {}).get(
                "decomposition_unresolved", False
            )
        )
    }
    if referenced_unresolved_decompositions != set(
        unresolved_decomposition_by_response
    ):
        raise AssertionError(
            "Prepared response metadata and unresolved decomposition ledger do not match."
        )
    partial_path.replace(prepared_path)
    warning_rows = []
    for response in prepared:
        for probes in response.probe_sets:
            warnings = probe_warnings(probes)
            if warnings:
                warning_rows.append({"factoid_id": probes.factoid_id, "warnings": list(warnings)})
    source_factoid_count = sum(
        int((item.metadata.get("preparation") or {}).get("source_factoid_count", 0))
        for item in prepared
    )
    unresolved_count = len(unresolved_by_factoid)
    unresolved_decomposition_count = len(unresolved_decomposition_by_response)
    manifest.update(
        {
            "status": (
                "complete_with_unresolved_preparation"
                if unresolved_count or unresolved_decomposition_count
                else "complete"
            ),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "prepared_sha256": file_sha256(prepared_path),
            "prepared_responses": len(prepared),
            "prepared_factoids": sum(len(item.probe_sets) for item in prepared),
            "source_factoids": source_factoid_count,
            "unresolved_mutations": unresolved_count,
            "unresolved_mutation_rate": (
                unresolved_count / source_factoid_count
                if source_factoid_count
                else 0.0
            ),
            "responses_without_valid_probes": [
                item.response_id for item in prepared if not item.probe_sets
            ],
            "unresolved_mutations_path": str(unresolved_path),
            "unresolved_mutations_sha256": (
                file_sha256(unresolved_path) if unresolved_path.exists() else None
            ),
            "unresolved_decompositions": unresolved_decomposition_count,
            "unresolved_decomposition_rate": (
                unresolved_decomposition_count / len(source_responses)
                if source_responses
                else 0.0
            ),
            "unresolved_decompositions_path": str(unresolved_decomposition_path),
            "unresolved_decompositions_sha256": (
                file_sha256(unresolved_decomposition_path)
                if unresolved_decomposition_path.exists()
                else None
            ),
            "probe_warning_count": len(warning_rows),
            "probe_warnings": warning_rows,
            "preparation_calls_sha256": file_sha256(call_path) if call_path.exists() else None,
        }
    )
    manifest.pop("last_error", None)
    atomic_write_json(manifest_path, manifest)
    return prepared_path


def _resource_summary(results: list[Any]) -> dict[str, Any]:
    buckets: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "policy_calls": 0,
            "executed_calls": 0,
            "cache_hits": 0,
            "latency_seconds": 0.0,
        }
    )
    for result in results:
        for factoid in result.factoids:
            for observation in factoid.observations:
                bucket = buckets[observation.phase + ":" + observation.backend]
                bucket["input_tokens"] += observation.usage.input_tokens
                bucket["output_tokens"] += observation.usage.output_tokens
                bucket["policy_calls"] += observation.usage.calls
                bucket["cache_hits"] += int(observation.cache_hit)
                bucket["executed_calls"] += 0 if observation.cache_hit else observation.usage.calls
                bucket["latency_seconds"] += observation.latency_seconds
    output: dict[str, Any] = {}
    for key, values in buckets.items():
        values["total_tokens"] = int(values["input_tokens"]) + int(values["output_tokens"])
        output[key] = dict(values)
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Ledger row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _resource_summary_rows(calls: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "policy_calls": 0,
            "executed_calls": 0,
            "cache_hits": 0,
            "latency_seconds": 0.0,
            "ollama_total_duration_seconds": 0.0,
            "ollama_load_duration_seconds": 0.0,
            "ollama_prompt_eval_duration_seconds": 0.0,
            "ollama_generation_duration_seconds": 0.0,
        }
    )
    for call in calls:
        bucket = buckets[str(call["phase"]) + ":" + str(call["backend"])]
        usage = call.get("usage") or {}
        bucket["input_tokens"] += int(usage.get("input_tokens", call.get("input_tokens", 0)))
        bucket["output_tokens"] += int(usage.get("output_tokens", call.get("output_tokens", 0)))
        bucket["policy_calls"] += int(usage.get("calls", call.get("calls", 1)))
        cache_hit = bool(call.get("cache_hit", False))
        bucket["cache_hits"] += int(cache_hit)
        bucket["executed_calls"] += 0 if cache_hit else int(
            usage.get("calls", call.get("calls", 1))
        )
        bucket["latency_seconds"] += float(call.get("latency_seconds", 0.0))
        metadata = call.get("provider_metadata") or {}
        bucket["ollama_total_duration_seconds"] += float(
            metadata.get("total_duration_ns", 0) or 0
        ) / 1_000_000_000.0
        bucket["ollama_load_duration_seconds"] += float(
            metadata.get("load_duration_ns", 0) or 0
        ) / 1_000_000_000.0
        bucket["ollama_prompt_eval_duration_seconds"] += float(
            metadata.get("prompt_eval_duration_ns", 0) or 0
        ) / 1_000_000_000.0
        bucket["ollama_generation_duration_seconds"] += float(
            metadata.get("eval_duration_ns", 0) or 0
        ) / 1_000_000_000.0
    output: dict[str, Any] = {}
    for key, values in buckets.items():
        values["total_tokens"] = int(values["input_tokens"]) + int(values["output_tokens"])
        output[key] = dict(values)
    return output


def run_from_config(config: dict[str, Any], *, repository_root: str | Path) -> Path:
    validate_run_config(config)
    root = Path(repository_root)
    prepared_path = root / str(config["prepared_path"])
    all_prepared = load_prepared_responses(prepared_path)
    excluded_zero_probe_responses = [
        item.response_id for item in all_prepared if not item.probe_sets
    ]
    prepared = [item for item in all_prepared if item.probe_sets]
    if not prepared:
        raise ValueError("Prepared dataset contains no response with a valid probe set.")
    stage1_config = config["stage1"]
    stage1_cache = None
    if stage1_config.get("cache_path"):
        stage1_cache = VerificationCache(root / str(stage1_config["cache_path"])).load()
    stage1_client = build_client(stage1_config)
    stage1 = EvidenceVerifier(
        stage1_client,
        temperature=float(stage1_config.get("temperature", 0.0)),
        cache=stage1_cache,
    )
    stage2_config = config.get("stage2")
    stage2 = None
    stage2_client = None
    if stage2_config is not None:
        stage2_cache = None
        if stage2_config.get("cache_path"):
            stage2_cache = VerificationCache(root / str(stage2_config["cache_path"])).load()
        stage2_client = build_client(stage2_config)
        stage2 = EvidenceVerifier(
            stage2_client,
            temperature=float(stage2_config.get("temperature", 0.0)),
            cache=stage2_cache,
        )
    engine = CanonicalCascade(
        stage1_verifier=stage1,
        stage2_verifier=stage2,
        routing_policy=RoutingPolicy(str(config["routing_policy"])),
        threshold=float(config.get("threshold", 0.5)),
        anomaly_count_threshold=int(config.get("anomaly_count_threshold", 2)),
    )

    created_at = datetime.now(timezone.utc)
    run_id = str(config.get("run_id") or created_at.strftime("%Y%m%dT%H%M%SZ"))
    run_directory = root / str(config["output_root"]) / run_id
    resume = bool(config.get("resume", False))
    ledger = RunLedger(run_directory, resume=resume)
    prepared_digest = file_sha256(prepared_path)
    protocol, protocol_sha256 = _reproducibility_protocol(root)
    manifest = {
        "run_id": run_id,
        "created_at_utc": created_at.isoformat(),
        "git_commit": _git_commit(root),
        "config_hash": config_hash(config),
        "prepared_path": str(prepared_path),
        "prepared_sha256": prepared_digest,
        "routing_policy": str(config["routing_policy"]),
        "threshold": float(config.get("threshold", 0.5)),
        "scoring_profile": config["scoring_profile"],
        "response_aggregation": config["response_aggregation"],
        "stage1": config["stage1"],
        "stage2": config.get("stage2"),
        "python": sys.version,
        "platform": platform.platform(),
        "prompt_bundle": prompt_manifest(),
        "rq3_taxonomy": taxonomy_manifest(),
        "reproducibility_protocol": protocol,
        "reproducibility_protocol_sha256": protocol_sha256,
        "method_spec_sha256": file_sha256(root / "docs" / "method_spec.md"),
        "environment": config.get("environment", {}),
        "pricing": config.get("pricing", {}),
        "source_prepared_responses": len(all_prepared),
        "expected_responses": len(prepared),
        "excluded_zero_probe_responses": excluded_zero_probe_responses,
        "status": "running",
    }
    manifest_path = run_directory / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Run manifest already exists: {manifest_path}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "config_hash",
            "prepared_sha256",
            "routing_policy",
            "reproducibility_protocol_sha256",
        ):
            if existing.get(key) != manifest.get(key):
                raise ValueError(f"Resume manifest mismatch for {key}.")
        manifest = existing
        manifest["status"] = "running"
        manifest["resumed_at_utc"] = created_at.isoformat()
    ledger.write_manifest(manifest)
    completed = ledger.completed_response_ids()
    expected_ids = {item.response_id for item in prepared}
    if not completed.issubset(expected_ids):
        raise ValueError("Run ledger contains response IDs absent from the prepared dataset.")
    try:
        if expected_ids.difference(completed):
            warmup_count = int(
                protocol.get("latency_protocol", {}).get(
                    "local_warmup_calls_per_model_session", 0
                )
            )
            warmup_path = run_directory / "warmup_calls.jsonl"
            session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            warmup_context = "Paris is the capital city of France."
            warmup_statement = "Paris is in France."
            warmup_user_prompt = VERIFIER_USER_TEMPLATE.format(
                context=warmup_context,
                statement=warmup_statement,
            )
            for stage_name, client, model_config in (
                ("stage1", stage1_client, stage1_config),
                ("stage2", stage2_client, stage2_config),
            ):
                if client is None or model_config is None or client.backend != "ollama":
                    continue
                for warmup_index in range(warmup_count):
                    response = client.complete(
                        system_prompt=VERIFIER_SYSTEM_PROMPT,
                        user_prompt=warmup_user_prompt,
                        temperature=float(model_config.get("temperature", 0.0)),
                        phase="warmup_excluded_from_metrics",
                    )
                    _append_jsonl(
                        warmup_path,
                        {
                            "phase": "warmup_excluded_from_metrics",
                            "stage": stage_name,
                            "session_id": session_id,
                            "warmup_index": warmup_index,
                            **_prompt_audit_fields(
                                VERIFIER_SYSTEM_PROMPT, warmup_user_prompt
                            ),
                            **_response_audit_fields(response),
                        },
                    )
        for response in prepared:
            if response.response_id in completed:
                continue
            result = engine.process_response(
                response_id=response.response_id,
                question_id=response.question_id,
                gold_target=response.gold_target,
                context=response.context,
                probe_sets=response.probe_sets,
                metadata={
                    **dict(response.metadata or {}),
                    "answer_pair_id": response.answer_pair_id,
                    "answer_role": response.answer_role,
                },
            )
            ledger.append_result(result)
            completed.add(response.response_id)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["completed_responses"] = len(completed)
        manifest["last_error"] = f"{type(error).__name__}: {error}"
        ledger.write_manifest(manifest)
        raise

    response_rows = _read_jsonl(ledger.responses_path)
    factoid_rows = _read_jsonl(ledger.factoids_path)
    call_rows = _read_jsonl(ledger.calls_path)
    if len(response_rows) != len(prepared):
        raise AssertionError("Completed response count does not match the prepared dataset.")
    summary = summarize_serialized(response_rows, factoid_rows)
    source_factoids = sum(
        int(
            (item.metadata.get("preparation") or {}).get(
                "source_factoid_count", len(item.probe_sets)
            )
        )
        for item in all_prepared
    )
    unresolved_mutations = sum(
        int((item.metadata.get("preparation") or {}).get("unresolved_mutation_count", 0))
        for item in all_prepared
    )
    unresolved_decompositions = sum(
        int(
            bool(
                (item.metadata.get("preparation") or {}).get(
                    "decomposition_unresolved", False
                )
            )
        )
        for item in all_prepared
    )
    summary["preparation_coverage"] = {
        "source_responses": len(all_prepared),
        "analyzed_responses": len(prepared),
        "excluded_zero_probe_responses": excluded_zero_probe_responses,
        "source_factoids": source_factoids,
        "analyzed_factoids": sum(len(item.probe_sets) for item in all_prepared),
        "unresolved_mutations": unresolved_mutations,
        "unresolved_mutation_rate": (
            unresolved_mutations / source_factoids if source_factoids else 0.0
        ),
        "unresolved_decompositions": unresolved_decompositions,
        "unresolved_decomposition_rate": (
            unresolved_decompositions / len(all_prepared) if all_prepared else 0.0
        ),
    }
    categories = sorted(
        {
            str((row.get("metadata") or {}).get("factoid_type"))
            for row in response_rows
            if (row.get("metadata") or {}).get("factoid_type") is not None
        }
    )
    category_summary: dict[str, Any] = {}
    for category in categories:
        subset_responses = [
            row
            for row in response_rows
            if str((row.get("metadata") or {}).get("factoid_type")) == category
        ]
        subset_ids = {str(row["response_id"]) for row in subset_responses}
        subset_factoids = [row for row in factoid_rows if str(row["response_id"]) in subset_ids]
        category_summary[category] = summarize_serialized(subset_responses, subset_factoids)
    if category_summary:
        summary["by_factoid_type"] = category_summary
    summary["resources"] = _resource_summary_rows(call_rows)
    routing_total = sum(float(row.get("routing_latency_seconds", 0.0)) for row in factoid_rows)
    summary["routing_latency"] = {
        "factoids": len(factoid_rows),
        "total_seconds": routing_total,
        "mean_seconds_per_factoid": routing_total / len(factoid_rows) if factoid_rows else 0.0,
    }
    ledger.write_summary(summary)
    manifest["status"] = "complete"
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_responses"] = len(response_rows)
    manifest["completed_factoids"] = len(factoid_rows)
    manifest["output_sha256"] = {
        "responses": file_sha256(ledger.responses_path),
        "factoids": file_sha256(ledger.factoids_path),
        "calls": file_sha256(ledger.calls_path),
        "summary": file_sha256(run_directory / "summary.json"),
        "warmup_calls": (
            file_sha256(run_directory / "warmup_calls.jsonl")
            if (run_directory / "warmup_calls.jsonl").exists()
            else None
        ),
    }
    manifest.pop("last_error", None)
    ledger.write_manifest(manifest)
    return run_directory
