

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config
from .api_client import ApiResult, parse_boolean
from .data_io import (
    append_jsonl,
    atomic_write_json,
    latest_success_by_id,
    normalize_records,
    stable_hash,
    validate_item,
)
from .prompts import (
    INJECT_SYSTEM_PROMPT,
    INJECT_USER_TEMPLATE,
    NO2_SYSTEM_PROMPT,
    NO2_USER_TEMPLATE,
    NO3_HALLUCINATION_SYSTEM_PROMPT,
    NO3_HALLUCINATION_USER_TEMPLATE,
    NO3_RIGHT_SYSTEM_PROMPT,
    NO3_RIGHT_USER_TEMPLATE,
    assert_prompts_locked,
    prompt_hashes,
)


STOP_WORDS = {
    "is", "the", "in", "and", "to", "a", "an", "of", "for", "with",
    "on", "at", "by", "from", "has", "have", "had", "it", "that",
    "this", "are", "was", "were", "as", "be",
}
TOKEN_PATTERN = r"\b[a-zA-Z]{2,}\b"
NO1_THRESHOLD = 0.4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _core_tokens(text: str) -> set[str]:
    words = re.findall(TOKEN_PATTERN, (text or "").lower())
    return {word for word in words if word not in STOP_WORDS}


def _save_three_way(
    run_dir: Path,
    prefix: str,
    clean: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    unresolved: List[Dict[str, Any]],
) -> None:
    atomic_write_json(run_dir / f"{prefix}_clean.json", clean)
    atomic_write_json(run_dir / f"{prefix}_rejected.json", rejected)
    atomic_write_json(run_dir / f"{prefix}_unresolved.json", unresolved)


def _attach_api_result(record: Dict[str, Any], result: ApiResult) -> None:
    record.update(
        {
            "api_ok": result.ok,
            "raw_output": result.text,
            "api_error": result.error,
            "response_id": result.response_id,
            "model_returned": result.returned_model,
            "system_fingerprint": result.system_fingerprint,
            "usage": result.usage or {},
            "api_attempts": result.attempts,
            "attempt_errors": result.attempt_errors,
        }
    )


def _request_record(
    *,
    stage: str,
    item: Dict[str, Any],
    prompt_hash: str,
    messages: List[Dict[str, str]],
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "id": item["id"],
        "item_hash": stable_hash(item),
        "timestamp_utc": _utc_now(),
        "model_requested": config.MODEL_NAME,
        "prompt_hash": prompt_hash,
        "parameters": parameters,
        "request_hash": stable_hash(
            {
                "model": config.MODEL_NAME,
                "messages": messages,
                "parameters": parameters,
            }
        ),
    }


def _materialize_boolean(
    data: List[Dict[str, Any]],
    latest: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    clean, rejected, unresolved = [], [], []
    for item in data:
        record = latest.get(str(item["id"]))
        if record is None or record.get("item_hash") != stable_hash(item):
            unresolved.append(item)
        elif record.get("decision") is True:
            clean.append(item)
        else:
            rejected.append(item)
    return clean, rejected, unresolved


def clean_no1(
    raw_records: List[Dict[str, Any]],
    dataset: str,
    run_dir: Path,
    test_limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """NO1：Unify five fields, and execute history 40% Core Word Coverage Rules。"""
    data = normalize_records(raw_records, dataset)
    if test_limit is not None:
        data = data[:test_limit]
    atomic_write_json(run_dir / "00_normalized.json", data)

    clean, rejected, ledger = [], [], []
    rule = {
        "token_pattern": TOKEN_PATTERN,
        "stop_words": sorted(STOP_WORDS),
        "threshold": NO1_THRESHOLD,
        "empty_answer_tokens": "keep",
    }
    rule_hash = stable_hash(rule)
    for item in data:
        answer_tokens = _core_tokens(item["right_answer"])
        context_tokens = _core_tokens(item["context"])
        overlap = answer_tokens.intersection(context_tokens)
        coverage = None if not answer_tokens else len(overlap) / len(answer_tokens)
        decision = not answer_tokens or coverage >= NO1_THRESHOLD
        ledger.append(
            {
                "stage": "no1",
                "id": item["id"],
                "item_hash": stable_hash(item),
                "rule_hash": rule_hash,
                "coverage_ratio": coverage,
                "answer_token_count": len(answer_tokens),
                "overlap_token_count": len(overlap),
                "decision": decision,
            }
        )
        (clean if decision else rejected).append(item)

    atomic_write_json(run_dir / "ledgers" / "no1.json", ledger)
    atomic_write_json(run_dir / "01_no1_clean.json", clean)
    atomic_write_json(run_dir / "01_no1_rejected.json", rejected)
    return clean, rejected


def clean_no2(
    data: List[Dict[str, Any]],
    client: Any,
    run_dir: Path,
    resume: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """NO2"""
    assert_prompts_locked()
    stage = "no2"
    prompt_hash = prompt_hashes()[stage]
    ledger_path = run_dir / "ledgers" / "no2.jsonl"
    latest = latest_success_by_id(ledger_path, stage, prompt_hash) if resume else {}
    parameters = {
        "temperature": 0.0,
        "max_tokens": config.BOOLEAN_MAX_TOKENS,
        "thinking_mode": config.THINKING_MODE,
    }

    def save() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        result = _materialize_boolean(data, latest)
        _save_three_way(run_dir, "02_no2", *result)
        return result

    completed_now = 0
    for position, item in enumerate(data, 1):
        existing = latest.get(str(item["id"]))
        if existing is not None and existing.get("item_hash") == stable_hash(item):
            continue

        messages = [
            {"role": "system", "content": NO2_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": NO2_USER_TEMPLATE.format(
                    context=item["context"],
                    right_answer=item["right_answer"],
                ),
            },
        ]
        record = _request_record(
            stage=stage,
            item=item,
            prompt_hash=prompt_hash,
            messages=messages,
            parameters=parameters,
        )
        api_result = client.complete(
            messages=messages,
            temperature=0.0,
            max_tokens=config.BOOLEAN_MAX_TOKENS,
        )
        _attach_api_result(record, api_result)
        decision = parse_boolean(api_result.text) if api_result.ok else None
        if decision is None:
            record.update({"status": "error", "decision": None, "reason": "api_or_parse_failure"})
        else:
            record.update({"status": "success", "decision": decision})
            latest[str(item["id"])] = record
            completed_now += 1
        append_jsonl(ledger_path, record)

        if completed_now and completed_now % config.CHECKPOINT_EVERY == 0:
            save()
            print(f"NO2 progress：{position}/{len(data)}")
        if config.REQUEST_INTERVAL_SECONDS:
            time.sleep(config.REQUEST_INTERVAL_SECONDS)
    return save()


def inject_hallucinations(
    data: List[Dict[str, Any]],
    client: Any,
    run_dir: Path,
    resume: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Hallucination Injection."""
    assert_prompts_locked()
    stage = "injection"
    prompt_hash = prompt_hashes()[stage]
    ledger_path = run_dir / "ledgers" / "injection.jsonl"
    latest = latest_success_by_id(ledger_path, stage, prompt_hash) if resume else {}
    parameters = {
        "temperature": 0.7,
        "max_tokens": config.INJECTION_MAX_TOKENS,
        "thinking_mode": config.THINKING_MODE,
        "max_generation_attempts": config.MAX_INJECTION_ATTEMPTS,
    }

    def save() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        injected, unresolved = [], []
        for item in data:
            record = latest.get(str(item["id"]))
            if record is None or record.get("item_hash") != stable_hash(item):
                unresolved.append(item)
            else:
                new_item = dict(item)
                new_item["hallucinated_answer"] = record["hallucinated_answer"]
                injected.append(new_item)
        atomic_write_json(run_dir / "03_injected.json", injected)
        atomic_write_json(run_dir / "03_injection_unresolved.json", unresolved)
        return injected, unresolved

    completed_now = 0
    for position, item in enumerate(data, 1):
        if item.get("hallucinated_answer"):
            raise ValueError("Hallucination injection function is only used for hallucinated_answer ASQA data is empty。")
        existing = latest.get(str(item["id"]))
        if existing is not None and existing.get("item_hash") == stable_hash(item):
            continue

        messages = [
            {"role": "system", "content": INJECT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": INJECT_USER_TEMPLATE.format(
                    question=item["question"],
                    context=item["context"],
                    right_answer=item["right_answer"],
                ),
            },
        ]
        success_record = None
        for generation_attempt in range(1, config.MAX_INJECTION_ATTEMPTS + 1):
            record = _request_record(
                stage=stage,
                item=item,
                prompt_hash=prompt_hash,
                messages=messages,
                parameters=parameters,
            )
            record["generation_attempt"] = generation_attempt
            api_result = client.complete(
                messages=messages,
                temperature=0.7,
                max_tokens=config.INJECTION_MAX_TOKENS,
            )
            _attach_api_result(record, api_result)
            generated = api_result.text.strip() if api_result.ok else ""
            if generated.startswith('"') and generated.endswith('"') and len(generated) >= 2:
                generated = generated[1:-1].strip()
            valid = bool(generated) and len(generated) >= 5 and generated != item["right_answer"]
            if valid:
                record.update(
                    {
                        "status": "success",
                        "hallucinated_answer": generated,
                        "generated_text_hash": stable_hash(generated),
                    }
                )
                append_jsonl(ledger_path, record)
                success_record = record
                break
            record.update(
                {
                    "status": "error",
                    "hallucinated_answer": generated,
                    "reason": "api_failure" if not api_result.ok else "empty_same_or_too_short",
                }
            )
            append_jsonl(ledger_path, record)

        if success_record is not None:
            latest[str(item["id"])] = success_record
            completed_now += 1
            if completed_now % config.CHECKPOINT_EVERY == 0:
                save()
                print(f"Hallucination injection progress:{position}/{len(data)}")
        if config.REQUEST_INTERVAL_SECONDS:
            time.sleep(config.REQUEST_INTERVAL_SECONDS)
    return save()


def _run_no3_boolean_call(
    *,
    stage: str,
    item: Dict[str, Any],
    ledger_path: Path,
    client: Any,
    system_prompt: str,
    user_prompt: str,
    prompt_hash: str,
) -> Optional[Dict[str, Any]]:
    """Execute and record a NO3 boolean check; return on parse failure None。"""
    parameters = {
        "temperature": 0.0,
        "max_tokens": config.BOOLEAN_MAX_TOKENS,
        "thinking_mode": config.THINKING_MODE,
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    record = _request_record(
        stage=stage,
        item=item,
        prompt_hash=prompt_hash,
        messages=messages,
        parameters=parameters,
    )
    api_result = client.complete(
        messages=messages,
        temperature=0.0,
        max_tokens=config.BOOLEAN_MAX_TOKENS,
    )
    _attach_api_result(record, api_result)
    decision = parse_boolean(api_result.text) if api_result.ok else None
    if decision is None:
        record.update(
            {"status": "error", "decision": None, "reason": "api_or_parse_failure"}
        )
        append_jsonl(ledger_path, record)
        return None
    record.update({"status": "success", "decision": decision})
    append_jsonl(ledger_path, record)
    return record


def clean_no3(
    data: List[Dict[str, Any]],
    client: Any,
    run_dir: Path,
    resume: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """ NO3：The correct answer is supported, and the hallucinated answer indeed contains hallucinations."""
    assert_prompts_locked()
    right_stage = "no3_right"
    hallucination_stage = "no3_hallucination"
    right_ledger = run_dir / "ledgers" / "no3_right.jsonl"
    hallucination_ledger = run_dir / "ledgers" / "no3_hallucination.jsonl"
    hashes = prompt_hashes()
    right_latest = (
        latest_success_by_id(right_ledger, right_stage, hashes[right_stage])
        if resume
        else {}
    )
    hallucination_latest = (
        latest_success_by_id(
            hallucination_ledger,
            hallucination_stage,
            hashes[hallucination_stage],
        )
        if resume
        else {}
    )

    def save() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        clean: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        unresolved: List[Dict[str, Any]] = []
        for item in data:
            key = str(item["id"])
            item_digest = stable_hash(item)
            right = right_latest.get(key)
            hallucination = hallucination_latest.get(key)
            if right is None or right.get("item_hash") != item_digest:
                unresolved.append(item)
            elif right.get("decision") is not True:
                rejected.append(item)
            elif (
                hallucination is None
                or hallucination.get("item_hash") != item_digest
            ):
                unresolved.append(item)
            elif hallucination.get("decision") is True:
                clean.append(item)
            else:
                rejected.append(item)
        result = (clean, rejected, unresolved)
        _save_three_way(run_dir, "04_no3", *result)
        return result

    completed_now = 0
    for position, item in enumerate(data, 1):
        errors = validate_item(item, require_hallucination=True)
        if errors:
            raise ValueError(f"NO3  id={item.get('id')} Illegal:{errors}")
        key = str(item["id"])
        item_digest = stable_hash(item)

        right = right_latest.get(key)
        if right is None or right.get("item_hash") != item_digest:
            right = _run_no3_boolean_call(
                stage=right_stage,
                item=item,
                ledger_path=right_ledger,
                client=client,
                system_prompt=NO3_RIGHT_SYSTEM_PROMPT,
                user_prompt=NO3_RIGHT_USER_TEMPLATE.format(
                    context=item["context"],
                    right_answer=item["right_answer"],
                ),
                prompt_hash=hashes[right_stage],
            )
            if right is not None:
                right_latest[key] = right
                completed_now += 1


        if right is None or right.get("decision") is not True:
            if completed_now and completed_now % config.CHECKPOINT_EVERY == 0:
                save()
                print(f"NO3 Progress:{position}/{len(data)}")
            if config.REQUEST_INTERVAL_SECONDS:
                time.sleep(config.REQUEST_INTERVAL_SECONDS)
            continue

        hallucination = hallucination_latest.get(key)
        if (
            hallucination is None
            or hallucination.get("item_hash") != item_digest
        ):
            hallucination = _run_no3_boolean_call(
                stage=hallucination_stage,
                item=item,
                ledger_path=hallucination_ledger,
                client=client,
                system_prompt=NO3_HALLUCINATION_SYSTEM_PROMPT,
                user_prompt=NO3_HALLUCINATION_USER_TEMPLATE.format(
                    context=item["context"],
                    hallucinated_answer=item["hallucinated_answer"],
                ),
                prompt_hash=hashes[hallucination_stage],
            )
            if hallucination is not None:
                hallucination_latest[key] = hallucination
                completed_now += 1

        if completed_now and completed_now % config.CHECKPOINT_EVERY == 0:
            save()
            print(f"NO3 Progress:{position}/{len(data)}")
        if config.REQUEST_INTERVAL_SECONDS:
            time.sleep(config.REQUEST_INTERVAL_SECONDS)
    return save()
