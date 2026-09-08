"""
First, select the dataset and TEST_LIMIT in `simple_cleaning/config.py`, then click Run.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from simple_cleaning import __version__, config
from simple_cleaning.api_client import DeepSeekClient
from simple_cleaning.data_io import atomic_write_json, file_sha256, load_json_or_jsonl, stable_hash
from simple_cleaning.prompts import PROMPTS, assert_prompts_locked, prompt_hashes
from simple_cleaning.steps import (
    NO1_THRESHOLD,
    STOP_WORDS,
    TOKEN_PATTERN,
    clean_no1,
    clean_no2,
    clean_no3,
    inject_hallucinations,
)


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _protocol_snapshot() -> Dict[str, Any]:
    no1_rule = {
        "token_pattern": TOKEN_PATTERN,
        "stop_words": sorted(STOP_WORDS),
        "threshold": NO1_THRESHOLD,
        "empty_answer_tokens": "keep",
    }
    return {
        "route": {
            "asqa": [
                "NO1",
                "NO2_original",
                "hallucination_injection_user_specified",
                "NO3_original_dual_validation",
            ],
            "haluevalqa": [
                "NO1",
                "NO2_original",
                "NO3_original_dual_validation",
            ],
        },
        "no1_rule": no1_rule,
        "no1_rule_hash": stable_hash(no1_rule),
        "prompts": PROMPTS,
        "prompt_hashes": prompt_hashes(),
        "no3_checks_right_answer": True,
    }


def _new_manifest(dataset: str, input_path: Path) -> Dict[str, Any]:
    protocol = _protocol_snapshot()
    return {
        "pipeline_version": __version__,
        "dataset": dataset,
        "created_utc": _utc_now(),
        "input_path": str(input_path.resolve()),
        "input_sha256": file_sha256(input_path),
        "model": config.MODEL_NAME,
        "thinking_mode": config.THINKING_MODE,
        "test_limit": config.TEST_LIMIT,
        "run_tag": config.RUN_TAG,
        "prompt_hashes": protocol["prompt_hashes"],
        "no1_rule_hash": protocol["no1_rule_hash"],
        "no3_checks_right_answer": True,
        "status": "initialized",
    }


def _prepare_manifest(dataset: str, input_path: Path, run_dir: Path) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_manifest.json"
    expected = _new_manifest(dataset, input_path)
    if path.exists():
        if not config.RESUME:
            raise RuntimeError(f"The working directory already exists：{run_dir}。Please change RUN_TAG, or set RESUME to True.")
        current = json.loads(path.read_text(encoding="utf-8"))
        locked_keys = (
            "pipeline_version",
            "dataset",
            "input_sha256",
            "model",
            "thinking_mode",
            "test_limit",
            "run_tag",
            "prompt_hashes",
            "no1_rule_hash",
            "no3_checks_right_answer",
        )
        mismatches = {
            key: {"old": current.get(key), "new": expected.get(key)}
            for key in locked_keys
            if current.get(key) != expected.get(key)
        }
        if mismatches:
            raise RuntimeError(f"The continuation run configuration is inconsistent, please change RUN_TAG.：{mismatches}")
        return current

    atomic_write_json(path, expected)
    atomic_write_json(run_dir / "protocol_snapshot.json", _protocol_snapshot())
    return expected


def _update_manifest(run_dir: Path, manifest: Dict[str, Any], **updates: Any) -> None:
    manifest.update(updates)
    manifest["updated_utc"] = _utc_now()
    atomic_write_json(run_dir / "run_manifest.json", manifest)


def main() -> int:
    _configure_utf8_console()
    config.validate_config()
    assert_prompts_locked()
    model_client: Optional[DeepSeekClient] = None

    for dataset in config.DATASETS_TO_RUN:
        input_path = config.INPUT_FILES[dataset]
        if not input_path.exists():
            raise FileNotFoundError(
                f"Cannot find {dataset} Original file：{input_path}\n"
                "If the code is not adjacent to the data folder, please set METARAG_DATA_DIR."
            )
        run_dir = config.OUTPUT_ROOT / f"{dataset}_{config.RUN_TAG}"
        manifest = _prepare_manifest(dataset, input_path, run_dir)
        print(f"\nStart processing{dataset}；Output Directory：{run_dir}")

        try:
            raw_records = load_json_or_jsonl(input_path)

            no1_clean, no1_rejected = clean_no1(
                raw_records,
                dataset,
                run_dir,
                test_limit=config.TEST_LIMIT,
            )
            _update_manifest(
                run_dir,
                manifest,
                status="no1_complete",
                raw_count=len(raw_records),
                no1_counts={"clean": len(no1_clean), "rejected": len(no1_rejected)},
            )
            print(f"NO1 lexical pre-filter ：Reserved {len(no1_clean)}，eliminate {len(no1_rejected)}")
            if config.DRY_RUN:
                _update_manifest(run_dir, manifest, status="dry_run_complete")
                continue

            if model_client is None:
                model_client = DeepSeekClient(
                    model=config.MODEL_NAME,
                    base_url=config.BASE_URL,
                    thinking_mode=config.THINKING_MODE,
                    max_retries=config.MAX_API_RETRIES,
                )

            no2_clean, no2_rejected, no2_unresolved = clean_no2(
                no1_clean,
                model_client,
                run_dir,
                resume=config.RESUME,
            )
            _update_manifest(
                run_dir,
                manifest,
                status="no2_complete" if not no2_unresolved else "no2_incomplete",
                no2_counts={
                    "clean": len(no2_clean),
                    "rejected": len(no2_rejected),
                    "unresolved": len(no2_unresolved),
                },
            )
            print(
                f"NO2 support check：Reserved {len(no2_clean)}，eliminate {len(no2_rejected)}，"
                f"Unresolved {len(no2_unresolved)}"
            )
            if no2_unresolved:
                raise RuntimeError("NO2 There are unresolved items; you can resume running after fixing the API/output issues.")

            if dataset == "asqa":
                no3_input, injection_unresolved = inject_hallucinations(
                    no2_clean,
                    model_client,
                    run_dir,
                    resume=config.RESUME,
                )
                _update_manifest(
                    run_dir,
                    manifest,
                    status=(
                        "injection_complete"
                        if not injection_unresolved
                        else "injection_incomplete"
                    ),
                    injection_counts={
                        "injected": len(no3_input),
                        "unresolved": len(injection_unresolved),
                    },
                )
                print(
                    f"Hallucination Injection: Complete {len(no3_input)}，Unresolved {len(injection_unresolved)}"
                )
                if injection_unresolved:
                    raise RuntimeError("There are unresolved items in the hallucination injection; rerun to continue.")
            else:
                no3_input = no2_clean

            final_clean, no3_rejected, no3_unresolved = clean_no3(
                no3_input,
                model_client,
                run_dir,
                resume=config.RESUME,
            )
            _update_manifest(
                run_dir,
                manifest,
                status="complete" if not no3_unresolved else "no3_incomplete",
                no3_counts={
                    "clean": len(final_clean),
                    "rejected": len(no3_rejected),
                    "unresolved": len(no3_unresolved),
                },
            )
            if no3_unresolved:
                raise RuntimeError("NO3 paired-answer validation There are unresolved items; you can resume by rerunning.")
            atomic_write_json(run_dir / "final_clean.json", final_clean)
            print(f"Completed:{dataset} Final reserve {len(final_clean)} Article.")
        except Exception as exc:
            _update_manifest(
                run_dir,
                manifest,
                status="incomplete_or_failed",
                last_error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
