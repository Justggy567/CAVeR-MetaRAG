"""Offline auditing of frozen data and the runtime environment."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .datasets import file_sha256
from .ledger import atomic_write_json
from .prompts import prompt_manifest
from .taxonomy import RQ3_CATEGORY_NAMES, taxonomy_manifest


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number} must contain a JSON object.")
                rows.append(value)
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"{path} must contain a JSON array of objects.")
    return value


def audit_dataset(
    path: str | Path,
    *,
    dataset_format: str,
    expected_count: int | None = None,
    expected_sha256: str | None = None,
    expected_category_counts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    digest = file_sha256(source)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch for {source}: {digest}")
    rows = _load_records(source)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Count mismatch for {source}: expected {expected_count}, got {len(rows)}")

    canonical_fields = (
        ("id", "question", "context", "right_answer", "hallucinated_answer")
        if dataset_format == "paired_answers"
        else ("id", "question", "context", "factoid", "category", "context_support")
    )
    empty_or_missing: list[dict[str, Any]] = []
    ids: list[str] = []
    for index, row in enumerate(rows):
        missing = [
            field
            for field in canonical_fields
            if field not in row or (isinstance(row.get(field), str) and not row[field].strip())
        ]
        if missing:
            empty_or_missing.append({"row": index, "id": row.get("id"), "fields": missing})
        ids.append(str(row.get("id", index)))
    if empty_or_missing:
        raise ValueError(f"Canonical field audit failed: {empty_or_missing[:10]}")
    duplicate_ids = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"Duplicate IDs: {duplicate_ids[:10]}")

    report: dict[str, Any] = {
        "path": str(source.resolve()),
        "sha256": digest,
        "format": dataset_format,
        "records": len(rows),
        "unique_ids": len(set(ids)),
        "canonical_fields": list(canonical_fields),
    }
    if dataset_format == "paired_answers":
        report.update(
            {
                "answer_pairs": len(rows),
                "evaluated_responses_if_both_roles": 2 * len(rows),
                "supported_responses_if_both_roles": len(rows),
                "unsupported_responses_if_both_roles": len(rows),
                "unique_questions": len({str(row["question"]).strip() for row in rows}),
            }
        )
    elif dataset_format == "factoids":
        category_counts: dict[str, dict[str, int]] = {}
        for category in sorted({str(row["category"]) for row in rows}):
            group = [row for row in rows if str(row["category"]) == category]
            category_counts[category] = {
                "n": len(group),
                "supported": sum(row["context_support"] is True for row in group),
                "unsupported": sum(row["context_support"] is False for row in group),
            }
        unmapped = sorted(set(category_counts).difference(RQ3_CATEGORY_NAMES))
        if unmapped:
            raise ValueError(f"RQ3 category codes missing from frozen taxonomy: {unmapped}")
        if expected_category_counts is not None and category_counts != expected_category_counts:
            raise ValueError(
                "RQ3 category balance mismatch: "
                f"expected {expected_category_counts}, got {category_counts}"
            )
        report.update(
            {
                "factoids": len(rows),
                "unique_factoids": len({str(row["factoid"]).strip() for row in rows}),
                "unique_source_questions": len({str(row.get("source_id", row["id"])) for row in rows}),
                "category_counts": category_counts,
                "category_name_mapping": {
                    code: RQ3_CATEGORY_NAMES.get(code, "unmapped")
                    for code in sorted(category_counts)
                },
            }
        )
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")
    return report


def audit_data_config(config: dict[str, Any], *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root)
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("audit config requires a non-empty datasets array.")
    reports = []
    for item in datasets:
        reports.append(
            audit_dataset(
                root / str(item["path"]),
                dataset_format=str(item["format"]),
                expected_count=int(item["expected_count"]) if item.get("expected_count") is not None else None,
                expected_sha256=str(item["expected_sha256"]) if item.get("expected_sha256") else None,
                expected_category_counts=(
                    dict(item["expected_category_counts"])
                    if item.get("expected_category_counts") is not None
                    else None
                ),
            )
        )
    return {
        "datasets": reports,
        "prompt_bundle": prompt_manifest(),
        "rq3_taxonomy": taxonomy_manifest(),
    }


def _capture(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": f"{type(error).__name__}: {error}"}


def environment_report() -> dict[str, Any]:

    optional_resources: dict[str, Any] = {}
    try:
        import cpuinfo

        cpu = cpuinfo.get_cpu_info()
        optional_resources["cpu"] = {
            key: cpu.get(key)
            for key in (
                "brand_raw",
                "vendor_id_raw",
                "arch",
                "bits",
                "count",
                "hz_advertised_friendly",
            )
        }
    except Exception as error:  
        optional_resources["cpu"] = {
            "status": "unavailable",
            "reason": f"{type(error).__name__}: {error}",
        }
    try:
        import psutil

        memory = psutil.virtual_memory()
        optional_resources["memory"] = {
            "total_bytes": int(memory.total),
            "total_gib": memory.total / (1024**3),
        }
        optional_resources["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        optional_resources["cpu_logical_cores"] = psutil.cpu_count(logical=True)
    except Exception as error:
        optional_resources["memory"] = {
            "status": "unavailable",
            "reason": f"{type(error).__name__}: {error}",
        }

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor_python": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "resource_snapshot": optional_resources,
        "cpu_windows_cim": _capture(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress",
            ]
        ),
        "memory_windows_cim": _capture(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory | ConvertTo-Json -Compress",
            ]
        ),
        "ollama_version": _capture(["ollama", "--version"]),
        "ollama_list": _capture(["ollama", "list"]),
        "ollama_models": {
            model: _capture(["ollama", "show", model])
            for model in ("gemma3:4b", "gemma3:12b", "phi4:latest")
        },
        "gpu": _capture(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "pip_freeze": _capture([sys.executable, "-m", "pip", "freeze"]),
        "energy_measurement": {
            "status": "unavailable",
            "reason": "No external power telemetry is collected by the canonical runner; GPU duration is not energy.",
        },
    }


def _model_parameter_row(config_path: Path, role: str, model: dict[str, Any]) -> dict[str, Any]:
    backend = str(model.get("backend", ""))
    common = ("model_id", "temperature", "top_p", "seed", "timeout_seconds")
    backend_specific = (
        ("endpoint", "top_k", "num_ctx", "num_predict", "keep_alive")
        if backend == "ollama"
        else (
            "base_url",
            "api_key_env",
            "max_tokens",
            "thinking_mode",
            "max_retries",
        )
        if backend == "openai_compatible"
        else ()
    )
    required = ("backend", *common, *backend_specific)
    missing = [name for name in required if name not in model]
    if missing:
        raise ValueError(
            f"Reviewer parameter audit failed for {config_path}:{role}; missing {missing}"
        )
    return {
        "config": str(config_path),
        "role": role,
        "backend": backend,
        **{name: model.get(name) for name in (*common, *backend_specific)},
        "api_key_value_recorded": False,
    }


def parameter_report(repository_root: str | Path) -> dict[str, Any]:
    """Check the configuration item by item to verify whether the model and decoding parameters are explicitly present."""

    root = Path(repository_root)
    config_paths = sorted(root.glob("configs/prepare_*.json"))
    config_paths.extend(sorted(root.glob("configs/rq1/**/*.json")))
    config_paths.extend(sorted(root.glob("configs/rq2/*.json")))
    config_paths.extend(sorted(root.glob("configs/rq3/*.json")))
    rows: list[dict[str, Any]] = []
    preparation_output_validation: list[dict[str, Any]] = []
    rq1_models: set[str] = set()
    rq1_configs = 0
    for path in config_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(root)
        if path.name.startswith("prepare_"):
            rows.append(
                _model_parameter_row(relative, "preparation_model", value["preparation_model"])
            )
            preparation_output_validation.append(
                {
                    "config": str(relative),
                    "decomposition_format_max_attempts": int(
                        value["decomposition_format_max_attempts"]
                    ),
                    "decomposition_failure_policy": str(
                        value["decomposition_failure_policy"]
                    ),
                    "mutation_temperature": float(value["mutation_temperature"]),
                    "mutation_format_max_attempts": int(
                        value["mutation_format_max_attempts"]
                    ),
                    "mutation_failure_policy": str(value["mutation_failure_policy"]),
                    "resume": bool(value["resume"]),
                    "required_synonyms": 2,
                    "required_antonyms": 2,
                    "retry_prompt": "identical frozen prompt",
                    "manual_repair_permitted": False,
                    "unresolved_mutations_reported_separately": True,
                    "unresolved_decompositions_reported_separately": True,
                }
            )
            continue
        if "rq1" in path.parts:
            rq1_configs += 1
            rq1_models.add(str(value["stage1"]["model_id"]))
        rows.append(_model_parameter_row(relative, "stage1", value["stage1"]))
        if value.get("stage2") is not None:
            rows.append(_model_parameter_row(relative, "stage2", value["stage2"]))

    expected_models = {
        "gemma3:4b",
        "gemma3:12b",
        "phi4:latest",
        "deepseek-v4-flash",
    }
    if rq1_models != expected_models or rq1_configs != 8:
        raise ValueError(
            "RQ1 design audit failed: expected four verifier models and eight run configurations; "
            f"got models={sorted(rq1_models)}, configs={rq1_configs}."
        )
    protocol_path = root / "configs" / "reproducibility_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    return {
        "status": "passed",
        "rq1_design": {
            "verifier_models": sorted(rq1_models),
            "verifier_model_count": len(rq1_models),
            "dataset_count": 2,
            "primary_run_config_count": rq1_configs,
            "clarification": "four models times two datasets equals eight runs; not eight models",
        },
        "prompt_bundle": prompt_manifest(),
        "rq3_taxonomy": taxonomy_manifest(),
        "reproducibility_protocol_path": str(protocol_path.resolve()),
        "reproducibility_protocol_sha256": file_sha256(protocol_path),
        "reproducibility_protocol": protocol,
        "model_parameter_rows": rows,
        "preparation_output_validation": preparation_output_validation,
        "runtime_return_fields": [
            "returned_model_id",
            "provider_response_id",
            "system_fingerprint",
            "finish_reason",
            "attempts",
            "input_tokens",
            "output_tokens",
            "latency_seconds",
            "provider_metadata",
        ],
        "unavailable_until_runtime": [
            "hosted provider model snapshot when the API returns no version/fingerprint",
            "actual API charge until explicit prices are supplied",
            "energy use without external power telemetry",
        ],
    }


def write_audit(path: str | Path, report: dict[str, Any]) -> None:
    atomic_write_json(Path(path), report)
