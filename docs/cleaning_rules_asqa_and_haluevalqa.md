Here is the complete, translated English version of the data cleaning rules document, maintaining the exact same logic, structure, and details as the original.

***

# ASQA / HaluEvalQA Data Cleaning Rules

ASQA (https://huggingface.co/datasets/Self-GRIT/asqa_eval) and HaluEvalQA (https://huggingface.co/datasets/pminervini/HaluEval/viewer/qa) use unified data cleaning principles. **All reserved samples must be verified based on the specified `context` as the sole factual basis, with no external knowledge allowed in the answer. For data containing hallucinatory answer fields, it must also be confirmed that `hallucinated_answer` contains facts that are unsupported or contradicted by the context.**

## 1. Unified Data Schema

Before cleaning, data from different sources is organized into a unified field structure:

| Field | Meaning |
| :--- | :--- |
| `context` | Context, evidence, or knowledge snippets serving as the basis for judgment. |
| `right_answer` | Expected correct answer supported by the context. |
| `hallucinated_answer` | Expected incorrect answer containing hallucinations (only applicable to datasets labeled with hallucinatory answers). |

For ASQA and HaluEvalQA, if the original field names differ from the schema above, field mapping must be performed first.

---

## 2. Stage 1: Lexical Coverage Pre-filter

The first step uses word coverage rules for rapid pre-screening to eliminate samples that are clearly "out of knowledge."

The specific rules are as follows:

1. Extract core English words from both `right_answer` and `context`.
2. Convert all words to lowercase.
3. Filter out common stopwords, such as `is`, `the`, `in`, `and`, `to`, `of`, etc.
4. Retain only English words with a length of at least two letters; pure numbers are excluded from calculations.
5. Use a set rather than word frequency statistics, meaning duplicate words are counted only once.
6. Calculate the coverage of the answer core words within the context core words:

The retention condition is:
[
   Core word coverage > 0.4
]

If core words cannot be extracted from `right_answer`, the sample is temporarily retained at this stage. If the coverage is below 0.4, the sample is classified as "out of knowledge" dirty data and discarded. This rule is implemented in the No01 script. Dataset cleaning records show:
* **ASQA:** Originally 948 items; 208 items discarded and 740 items retained under this rule.
* **HaluEvalQA:** Originally 10,000 items; 633 items discarded and 9,367 items retained under this rule.

---

## 3. Stage 2: LLM-Based Grounding Validation for `right_answer`

The second step utilizes DeepSeek as a judge model to further evaluate whether `right_answer` is fully supported by `context`.

The validation rules are as follows:

1. Judgments must be based solely on `context`; no external knowledge is allowed.
2. If all facts in `right_answer` are directly sourced from or can be directly inferred from `context`, it is classified as clean.
3. Synonymous paraphrasing and sentence restructuring are permitted.
4. If `right_answer` contains any unsupported names, locations, dates, numbers, entities, causal relationships, events, or other facts not present in `context`, it is classified as dirty.
5. If `right_answer` contradicts `context`, it is classified as dirty.
6. The model is required to output only `True` or `False`.

Using `deepseek-chat` with `temperature=0.0` and `max_tokens=5`, the model is prompted to output only a boolean judgment. If either `context` or `right_answer` is empty, the sample is directly moved to the dirty file.

* **HaluEvalQA:** Using the 9,367 samples from Stage 1 as input, 8,939 samples were retained and 428 samples were discarded.
* **ASQA:** Using the 740 samples from Stage 1 as input, 96 samples were retained and 644 samples were discarded.

---

## 4. Stage 3: Dual Validation of `right_answer` and `hallucinated_answer`

The third step is a more rigorous dual validation to ensure that samples simultaneously meet two conditions.

A sample is retained only when both of the following conditions are met:

1. `right_answer` is fully supported by `context`;
2. `hallucinated_answer` contains at least one fact that is unsupported by or contradicts `context`.

If `right_answer` is not clean, the sample is discarded. If `hallucinated_answer` is actually supported by `context`, it does not qualify as a valid hallucinated answer, and the sample is discarded. Samples with missing or empty fields, repeated API failures, or anomalous model outputs are also discarded.

### 4.1 `right_answer` Validation

The condition for retaining `right_answer` is that every factual claim must be explicitly supported by or directly inferable from `context`. If any unsupported or contradictory facts occur—including unsupported people, locations, dates, numbers, entities, causal relationships, or events—the answer is classified as dirty.

### 4.2 `hallucinated_answer` Validation

The condition for `hallucinated_answer` to be deemed a valid hallucinated answer is that it must contain at least one fact unsupported by `context`, or at least one fact that contradicts `context`. If all facts in `hallucinated_answer` can be supported by `context`, the sample is discarded because it is not a genuine hallucinated sample.

### 4.3 Missing-field and Failure Handling

If any key field among `context`, `right_answer`, or `hallucinated_answer` is empty, the sample is directly discarded.

Only standard `True` or `False` outputs are accepted. If a model output is anomalous or an API call fails, up to 3 retries are performed. After multiple consecutive failures, the sample is classified as dirty for subsequent manual review.