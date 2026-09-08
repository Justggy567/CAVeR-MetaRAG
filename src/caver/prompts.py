

from __future__ import annotations

import hashlib
import json


PROMPT_BUNDLE_VERSION = "2026-08-22.v1"


DECOMPOSITION_SYSTEM_PROMPT = """
You are a fact extraction assistant.
Your task is to extract all specific factual propositions from the given text.

Instructions:
1. Extract every distinct factual statement present in the input, even if the statement is incorrect, ambiguous, or nonsensical.
2. Each extracted proposition must be a complete, standalone sentence.
3. Each sentence must express only one atomic fact. (An atomic fact cannot be split into simpler factual statements.)
4. If a sentence contains multiple facts, split them into multiple atomic fact sentences.
5. Do not paraphrase, rewrite, summarize, interpret, infer, or judge any part of the input. Only extract and restate what is explicitly written.
6. Do not omit or correct any statements, regardless of their factual accuracy.
7. Output your answer as a JSON array of strings, with each element being one atomic factual sentence.
8. CRITICAL: Resolve all pronouns (he, she, it, they, this) to their exact entity names based on the context. Do not use pronouns in the extracted facts.

Example:
Input: Marie Curie discovered polonium and radium, and Albert Einstein developed the theory of relativity in 1905.
Output: ["Marie Curie discovered polonium.", "Marie Curie discovered radium.", "Albert Einstein developed the theory of relativity in 1905."]

Now, extract atomic facts from this text:
""".strip()

DECOMPOSITION_USER_TEMPLATE = "Input:\n{answer}"


SYNONYM_SYSTEM_PROMPT = """
You will be given a question and a factual answer (factoid).
Your task is to generate 2 synonyms (paraphrased statements with the same meaning) of the factoid, based on the context of the question.

Instructions:
- Each output must be a single atomic factual claim (cannot be split into smaller facts).
- Use only information explicitly present in the question or factoid. Do not invent, infer, or add external knowledge.
- A correct synonym is a statement that means exactly the same thing as the factoid, even if the wording is different.
- Do not output partial phrases, keywords, or combine/split facts.
- Each synonym must be a complete, grammatically correct sentence.
- Just return the sentences, one per line, without numbers, bullets, or any other output.

Example:
Question: Where was Einstein born?
Factoid: Einstein was born in Germany.
Good Synonym: Germany is the country where Einstein was born.
Bad Synonym: Einstein visited Germany. (not equivalent)
Bad Synonym: Einstein was born. (incomplete)
""".strip()


ANTONYM_SYSTEM_PROMPT = """
You will be given a question and a factual answer (factoid).
Your task is to generate 2 negations (contradictory statements) of the factoid, based on the context of the question.

Instructions:
- Each negation must directly contradict the factoid, focusing on what the question asks.
- Do not add new information not present in the factoid or question.
- Do not use double negations or wording that preserves the original meaning.
- Each negation must be a meaningful, grammatically correct sentence.
- Do not introduce unrelated facts.
- Ensure that each negation is relevant to the question's context.
- Just return the sentences, one per line, without numbers or bullets, and nothing else.

Example:
Question: Where was Einstein born?
Factoid: Einstein was born in Germany.
Good Antonym: Einstein was not born in Germany.
Bad Antonym: Einstein visited Germany. (not a contradiction)
Bad Antonym: Einstein was born in Austria. (adds new information)
Bad Antonym: Einstein was not not born in Germany. (double negation)
Bad Antonym: Was not born in Germany. (missing subject)
""".strip()

MUTATION_USER_TEMPLATE = "Input:\nQuestion: {question}\nFactoid: {factoid}"



VERIFIER_SYSTEM_PROMPT = """
You will be given a statement and passages that represent the ground truth.
Determine if the statement is supported by the passages, either explicitly or through clear implication.

Answer with one of the following only:
- YES: if the statement is clearly and completely supported by the passages.
- NO: if the statement is contradicted or directly refuted by the passages.
- NOT SURE: if the passages do not contain enough information to confirm or deny the statement.

Respond with YES, NO, or NOT SURE. Then, in one short sentence, explain the reason for your answer.

Examples:

Passages (Ground Truth): "Alice was born in Paris and moved to New York at the age of five."
Statement: "Alice spent her early childhood in France."
Answer: YES. The passage states Alice was born in Paris, which is in France.

Passages (Ground Truth): "Bob has never visited Japan but plans to travel there next summer."
Statement: "Bob visited Japan last year."
Answer: NO. The passage says Bob has never visited Japan.

Passages (Ground Truth): "Carol enjoys outdoor activities like hiking and cycling."
Statement: "Carol loves swimming."
Answer: NOT SURE. There is no information in the passages about Carol and swimming.
""".strip()

VERIFIER_USER_TEMPLATE = """Now, perform the task:
Passages (Ground Truth):
{context}

Statement:
{statement}

Answer:"""


def prompt_hash(text: str) -> str:
   

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_manifest() -> dict[str, object]:
    

    prompts = {
        "decomposition_system": DECOMPOSITION_SYSTEM_PROMPT,
        "decomposition_user_template": DECOMPOSITION_USER_TEMPLATE,
        "synonym_system": SYNONYM_SYSTEM_PROMPT,
        "antonym_system": ANTONYM_SYSTEM_PROMPT,
        "mutation_user_template": MUTATION_USER_TEMPLATE,
        "verifier_system": VERIFIER_SYSTEM_PROMPT,
        "verifier_user_template": VERIFIER_USER_TEMPLATE,
    }
    hashes = {name: prompt_hash(value) for name, value in prompts.items()}
    bundle_payload = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
    return {
        "version": PROMPT_BUNDLE_VERSION,
        "hashes": hashes,
        "bundle_hash": prompt_hash(bundle_payload),
    }
