




# 96 ASQA Sample Selection and Hallucinated-Answer Injection

## 1. Purpose 

The ASQA subset used in this project was constructed to provide a small but strictly grounded QA benchmark for hallucination detection. Since the original ASQA evaluation data does not contain a paired `hallucinated_answer` field, we first selected ASQA examples whose reference answers were fully grounded in the retrieved context, and then injected a synthetic hallucinated answer for each retained example.


1. **Clean answer selection**：first_select_them `standard_answer` / `right_answer` asqa_samples_supported_by_context；
2. **Hallucinated-answer injection**：based_on_each_item clean The ASQA sample generates a plausible but incorrect `hallucinated_answer`。

the_file_contains_a_unified_format

```json
{
  "id": "...",
  "question": "...",
  "context": "...",
  "right_answer": "...",
  "hallucinated_answer": "..."
}
```

---

## 2. Source data 

The ASQA candidate pool was derived from the like ALCE ASQA evaluation data. The preprocessing script downloads `asqa_eval.tar` from the Hugging Face dataset repository:
https://huggingface.co/datasets/Self-GRIT/asqa_eval
```text
https://huggingface.co/datasets/Self-GRIT/asqa_eval
```


---

## 3. Conversion to unified ASQA format 

The original ASQA examples were converted into an internal format before cleaning. For each raw example, the script retains the question, constructs a context from the top retrieved documents, extracts the annotated long answer, and preserves the original question ID when available.

the_specific_field_mapping_is_as_follows：

| Internal field    | Source / construction rule                                                    |
| ----------------- | ----------------------------------------------------------------------------- |
| `question`        | `item["question"]`                                                            |
| `docs_context`    | Concatenation of the top-3 retrieved document texts, i.e., `item["docs"][:3]` |
| `standard_answer` | `item["annotations"][0]["long_answer"]`, if available                         |
| `id`              | `item["question_id"]`, or the sequential index if no question ID exists       |

The script uses the text fields from the top-3 retrieved documents as the evidence context and joins them with blank lines. It also extracts the first annotated `long_answer` as the standard reference answer. 

---

## 4. Rule-based grounding pre-filter 

After format conversion, we applied a rule-based lexical grounding filter to remove examples whose reference answers were weakly connected to the retrieved context.

The filter extracts core English tokens from both the answer and the context by:

1. lowercasing all text;
2. extracting alphabetic English words with length at least two;
3. removing common stop words;
4. ignoring purely numeric tokens.

A sample is retained only if:

[
   coverage_rate_greater_than_40
]

If no core answer tokens can be extracted, the sample is temporarily retained at this stage.

The original ASQA candidate pool contains 948 samples. After the 40% word coverage rule, 208 entries were deemed "disconnected" and removed from dirty data, leaving 740 entries.

---

## 5. LLM-based grounding validation 

The rule-based filter is only a coarse pre-filter. We then used an LLM judge to verify whether each `standard_answer` was fully grounded in the provided context.

In this stage, the ASQA fields are conceptually mapped as:

| ASQA field        | Cleaning-script field   |
| ----------------- | ----------------------- |
| `docs_context`    | `knowledge` / `context` |
| `standard_answer` | `right_answer`          |

The LLM judge was instructed to output `True` only if the answer was entirely extracted from, summarized from, or directly supported by the context. If the answer introduced even one unsupported person, place, time, number, entity, relation, event, or other factual claim, the sample was discarded. Synonyms and structural paraphrases were allowed only when no new factual information was introduced. The judge used `deepseek-chat` with `temperature=0.0` and was required to output only `True` or `False`. 

A sample was discarded if either the context or the answer was empty. Unexpected model outputs were treated as `False`, i.e., not clean. 

After this grounding-validation stage, only **96 ASQA examples** were retained as clean ASQA examples. These 96 examples formed the clean seed set for the following hallucinated-answer injection step.



---

## 6. Hallucinated-answer injection 

Since the original ASQA data does not provide `hallucinated_answer`, we injected one hallucinated answer for each of the 96 clean ASQA examples.


This means the hallucination generation step is applied **after** the clean-answer filtering stage. 

For each clean ASQA example, the script reads:

```python
question = item.get("question", "")
context = item.get("context", "")
standard_answer = item.get("right_answer", "")
item_id = item.get("id", "")
```

Then it generates a synthetic hallucinated answer using `deepseek-chat`. The generated example is stored in the final unified format:

```python
new_item = {
    "id": item_id,
    "question": question,
    "context": context,
    "right_answer": standard_answer,
    "hallucinated_answer": hallucinated_ans
}
```

---

## 7. Hallucination generation strategy 

The hallucination generator was prompted to take the correct `standard_answer` and subtly modify it into a highly plausible but incorrect answer. It was instructed to preserve the original answer’s language, tone, sentence structure, and approximate length, while making one or more factual changes that are wrong or not supported by the context. 

The prompt randomly selects and combines 1 to 3 hallucination strategies from the following list:

| Strategy    | Description                                                                              |
| ----------- | ---------------------------------------------------------------------------------------- |
| Entity      | Replace key entities such as names, locations, organizations, aliases, or specific nouns |
| Negation    | Insert, remove, or reverse negative words or exceptions                                  |
| Quantifier  | Alter scope, range, or boundary expressions                                              |
| Time        | Modify dates, temporal markers, sequences, or chronological order                        |
| Comparative | Reverse or alter comparative or superlative relations                                    |
| Numerical   | Change numbers, units, percentages, or mathematical outcomes                             |
| Relational  | Swap roles, actions, causal dependencies, or relations between entities                  |

The model was required to output only the final hallucinated text, without explanations, strategy names, quotes, or extra prefixes. 

Generation was performed with:

```text
model = deepseek-chat
temperature = 0.7
max_tokens = 256
```

The higher temperature was used to allow more diverse plausible factual perturbations. 

---


## 8. Final ASQA subset 

The final ASQA benchmark file contains 96 clean ASQA examples with injected hallucinated answers. Each example includes:

| Field                 | Meaning                                                     |
| --------------------- | ----------------------------------------------------------- |
| `id`                  | Original ASQA question ID or fallback index                 |
| `question`            | ASQA question                                               |
| `context`             | Top-3 retrieved document context                            |
| `right_answer`        | Grounded reference answer                                   |
| `hallucinated_answer` | Synthetic hallucinated answer generated from `right_answer` |

The final dataset should be treated as a **fixed released subset** for reproducibility. Users should not regenerate the 96 examples unless they also use the same intermediate clean file and the same generation settings.

---

