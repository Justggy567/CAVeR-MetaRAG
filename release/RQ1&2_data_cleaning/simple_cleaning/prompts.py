"""Locked data cleaning prompt.


"""

from __future__ import annotations

import hashlib
from typing import Dict


NO2_SYSTEM_PROMPT = """You are a rigorous data cleaning expert. Your task is to verify whether the provided [Answer] is entirely grounded in the given [Context].

Rules:
1. If the [Answer] contains even a single person, place, time, specific number, or fact that is NOT mentioned in the [Context], it means external knowledge was introduced. You must output "False".
2. If the [Answer] is entirely extracted or summarized from the [Context] (synonyms and structural rephrasing are allowed as long as no new facts are added), you must output "True".
3. You MUST output ONLY the word "True" or "False". Do not output any other explanations, analysis, or punctuation."""

NO2_USER_TEMPLATE = "[Context]:\n{context}\n\n[Answer]:\n{right_answer}\n\nJudgment (True or False only):"


INJECT_SYSTEM_PROMPT = """You are an expert at generating evaluation data for AI hallucination detection. Your task is to take a correct [Standard Answer] and subtly modify it to create a highly plausible but incorrect [Hallucinated Answer] according to the given [Context].

## General Rules

1. Read the [Context] and the [Standard Answer] carefully.

2. Randomly select 1 to 3 categories from the "Hallucination Categories" listed below. The selected categories should be applicable to the factual content of the [Standard Answer].

3. Apply the selected categories to modify one or multiple factual parts of the [Standard Answer].

4. Every modification must introduce information that is either:

   * explicitly contradicted by the [Context], or
   * not supported by the [Context].

5. Do not introduce unrelated content. The hallucinated answer should remain semantically close to the original [Standard Answer] except for the intentionally modified factual information.

6. Preserve the original language, writing style, tone, sentence organization, and approximate length as closely as possible.

7. Make the hallucinated information subtle, plausible, confident, and grammatically natural. Avoid obviously absurd or unrealistic modifications unless required by the selected category.

8. Different selected categories may be applied to different factual claims in the answer, or combined within the same factual claim when appropriate.

9. Do not add explanations, uncertainty expressions, warnings, or indications that the answer has been modified.

10. Use only the [Context] and [Standard Answer] when constructing the hallucination. Do not rely on external knowledge to determine whether the injected information is correct.

## Hallucination Categories

Randomly select and apply 1 to 3 of the following categories:

1. Entity / Alias / Lexical Rewrite
   Change the identity, name, alias, category, membership, or lexical identity description of an entity, and the modification must actually alter the factual meaning. Pure synonym-based rewriting does not count as a hallucination.

2. Negation / Exception
   Change the polarity of a proposition, exclusion relationships, exceptions, permission/prohibition, existence/non-existence, or other related semantic relations.

3. Quantifier / Boundary
   Change the scope, lower or upper bounds, degree of coverage, frequency, or quantification/boundary relations such as *all*, *some*, *only*, *at least*, and *at most*.

4. Time / Sequence
   Change dates, years, durations, temporal positions, event order, or temporal relations such as *before* and *after*.

5. Comparative / Superlative
   Change comparative relations, rankings, relative magnitude, or superlative relations such as highest/lowest or largest/smallest.

6. Numerical / Computation
   Use this category only when the numerical value, quantity, measurement, proportion, unit, or computation result itself is the core factual element being modified. Do not select this category merely because a sentence contains numbers. Temporal numbers should preferentially be categorized as **Time / Sequence**, boundary-related numbers as **Quantifier / Boundary**, and numbers involved in comparative relations as **Comparative / Superlative**.

7. Multi-entity / Multi-hop
   Change the roles, relations, interactions, or dependencies among multiple entities, or alter a conclusion that can only be established by combining multiple contextual facts. The mere presence of two entity names is not sufficient for this category.

## Output Requirement

You MUST output ONLY the final [Hallucinated Answer].

Do not output:

the selected hallucination categories,
explanations,
analysis,
prefixes,
labels,
quotation marks,
or any conversational filler."""

INJECT_USER_TEMPLATE = """[Question]:
{question}

[Context]:
{context}

[Standard Answer]:
{right_answer}

[Hallucinated Answer]:"""


NO3_RIGHT_SYSTEM_PROMPT = """
You are a rigorous data cleaning expert. Your task is to verify whether the provided [Right Answer] is entirely grounded in the given [Context].

Rules:
1. Output "True" only if every factual claim in [Right Answer] is explicitly supported by, or can be directly inferred from, [Context].
2. Output "False" if [Right Answer] contains even one unsupported or contradictory factual claim, including unsupported people, places, dates, numbers, entities, causes, relations, or events.
3. Synonyms and structural paraphrases are allowed, but adding new factual information is not allowed.
4. Do not use external knowledge. Judge only according to [Context].
5. You MUST output ONLY the word "True" or "False". Do not output explanations, punctuation, or analysis.
""".strip()

NO3_RIGHT_USER_TEMPLATE = """
[Context]:
{context}

[Right Answer]:
{right_answer}

Judgment:
""".strip()


NO3_HALLUCINATION_SYSTEM_PROMPT = """
You are a rigorous hallucination-label validator for a QA dataset. Your task is to verify whether the provided [Hallucinated Answer] is actually hallucinated with respect to the given [Context].

Definition:
A [Hallucinated Answer] is actually hallucinated if it contains at least one factual claim that is not supported by [Context], or at least one factual claim that contradicts [Context].

Rules:
1. Output "True" if [Hallucinated Answer] contains one or more unsupported or contradictory factual claims.
2. Output "False" if every factual claim in [Hallucinated Answer] is supported by, or can be directly inferred from, [Context]. In this case, the answer is not truly hallucinated and should be discarded from a hallucination dataset.
3. Unsupported people, places, dates, numbers, entities, causes, relations, or events count as hallucinations.
4. Synonyms and structural paraphrases do NOT count as hallucinations if the meaning is fully supported by [Context].
5. Do not use external knowledge. Judge only according to [Context].
6. You MUST output ONLY the word "True" or "False". Do not output explanations, punctuation, or analysis.
""".strip()

NO3_HALLUCINATION_USER_TEMPLATE = """
[Context]:
{context}

[Hallucinated Answer]:
{hallucinated_answer}

Judgment:
""".strip()


PROMPTS: Dict[str, Dict[str, str]] = {
    "no2": {"system": NO2_SYSTEM_PROMPT, "user_template": NO2_USER_TEMPLATE},
    "injection": {
        "system": INJECT_SYSTEM_PROMPT,
        "user_template": INJECT_USER_TEMPLATE,
    },
    "no3_right": {
        "system": NO3_RIGHT_SYSTEM_PROMPT,
        "user_template": NO3_RIGHT_USER_TEMPLATE,
    },
    "no3_hallucination": {
        "system": NO3_HALLUCINATION_SYSTEM_PROMPT,
        "user_template": NO3_HALLUCINATION_USER_TEMPLATE,
    },
}


def prompt_hash(system_prompt: str, user_template: str) -> str:
    payload = f"SYSTEM\0{system_prompt}\0USER_TEMPLATE\0{user_template}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_hashes() -> Dict[str, str]:
    return {
        name: prompt_hash(parts["system"], parts["user_template"])
        for name, parts in PROMPTS.items()
    }


EXPECTED_PROMPT_HASHES: Dict[str, str] = {
    "no2": "81df1dec2e503fe64b80741ba012938b68809e6d500a032981091fb800775760",
    "injection": "ff0654efba4b06c34f510ba704c9977051ea9f23e82a44b26e74c0466292db0a",
    "no3_right": "e2c467410219ad5f817842c39ced1f2a4497ad7319ae2dcb8dd369991ddd3401",
    "no3_hallucination": "7f22f7f08dcc9aaaec0473d5e3cbcbe6e59f25bd5169712b07cdc95088b68e64",
}


def assert_prompts_locked() -> None:
    actual = prompt_hashes()
    if actual != EXPECTED_PROMPT_HASHES:
        raise RuntimeError(
            "The prompt has undergone character-level changes. If modification is necessary, please change the RUN_TAG and re-lock the hash."
            f" actual={actual}"
        )
