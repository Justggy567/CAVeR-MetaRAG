"""RQ3 New version of the seven-category taxonomy's unique code source."""

from __future__ import annotations

import hashlib
import json


RQ3_TAXONOMY_VERSION = "2026-08-22.user-v2"
RQ3_TAXONOMY_SOURCE_SHA256 = (
    "0f38475b49193009c1dcae932b893fcf7d153a275778089a5420f58f665edaf7"
)

RQ3_CATEGORY_NAMES = {
    "Cat1": "Entity / Alias / Lexical Relation",
    "Cat2": "Negation / Exception",
    "Cat3": "Quantifier / Boundary",
    "Cat4": "Time / Sequence",
    "Cat5": "Comparative / Superlative",
    "Cat6": "Numerical / Computation",
    "Cat7": "Multi-entity / Multi-hop",
}

RQ3_CATEGORY_DEFINITIONS_EN = {
    "Cat1": (
        "Verification primarily requires identifying an entity, alias, category, "
        "membership, attribute, or direct lexical/semantic relation. Pure synonym-based "
        "surface variation does not constitute a factual difference."
    ),
    "Cat2": (
        "Truth primarily depends on proposition polarity, exclusion, exception, "
        "permission/prohibition, existence/non-existence, or another explicitly negative "
        "or exception-based semantic relation."
    ),
    "Cat3": (
        "Verification primarily depends on scope, lower or upper bounds, degree of coverage, "
        "frequency, or quantification/boundary relations such as all, some, only, at least, "
        "and at most."
    ),
    "Cat4": (
        "Verification primarily depends on dates, years, durations, temporal positions, "
        "event order, or temporal relations such as before and after."
    ),
    "Cat5": (
        "Verification primarily requires establishing a comparative relation, ranking, "
        "relative magnitude, or a superlative relation such as highest/lowest or "
        "largest/smallest."
    ),
    "Cat6": (
        "A numerical value, quantity, measurement, proportion, unit, or computation result "
        "must itself be the principal factual element, or verification must require an "
        "explicit numerical operation such as counting, arithmetic, ratio calculation, or "
        "unit conversion. The mere presence of a number is insufficient. Temporal numbers "
        "take Time / Sequence priority, boundary quantities take Quantifier / Boundary "
        "priority, and comparison-dependent numbers take Comparative / Superlative priority."
    ),
    "Cat7": (
        "Verification requires combining relations among multiple entities, integrating two "
        "or more distinct contextual facts, or following a multi-step relational chain. "
        "Multiple entity names alone are insufficient; two or more evidence relations or "
        "contextual facts must be combined."
    ),
}


def taxonomy_manifest() -> dict[str, object]:
    payload = {
        "version": RQ3_TAXONOMY_VERSION,
        "source_sha256": RQ3_TAXONOMY_SOURCE_SHA256,
        "assignment_principle": (
            "Assign one primary category by the principal reasoning operation required to "
            "verify the factoid against context, not by surface lexical cues."
        ),
        "category_names": RQ3_CATEGORY_NAMES,
        "definitions_en": RQ3_CATEGORY_DEFINITIONS_EN,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {**payload, "manifest_sha256": hashlib.sha256(encoded).hexdigest()}

