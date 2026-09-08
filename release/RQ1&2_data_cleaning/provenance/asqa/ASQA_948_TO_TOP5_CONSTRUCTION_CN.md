# From 948 ASQA Source Records to the Top-5 Input Set



## Origin of the 948 Records

- Upstream dataset: `Self-GRIT/asqa_eval`
- Upstream split: `valid` (normalized to `validation` in some dataset versions)
- Record count: **948 records** in total for this specific revision and split.
- Sampling: **None**. The figure of 948 is not a random sample from a larger set but represents the complete record count for this specific revision and split.


## Construction of the Top-5 Set

The following deterministic transformation is applied to each of the 948 source records:

1. `question`: Taken directly from the source record's `question` field.
2. `standard_answer`: Taken from `annotations[0].long_answer`.
3. `docs_context`: Formed by taking the first five documents (`docs[:5]`) based on the existing upstream retrieval order, extracting their `text` fields, and concatenating them sequentially using two newline characters.
4. `id`: Uses the zero-based row index from the source split (range: 0–947).
5. No re-retrieval, re-ranking, or deduplication is performed; nor are documents selected based on correctness or hallucination labels. The output structure is fixed as follows:

```json
{
"question": "...",
"docs_context": "top-1\n\ntop-2\n\n...\n\ntop-5",
"standard_answer": "...",
"id": 0
}
```

## Reproduction Commands

Execute the following in the `RQ1&2_data_cleaning` directory:

```powershell
python -m pip install -r .\provenance\asqa\requirements_asqa_top5.txt
python .\provenance\asqa\build_asqa_top5.py --output .\reproduced\asqa_eval_data_top5.json
```

The script generates the following simultaneously:

- `asqa_eval_data_top5.json`: 948 records of Top-5 data (four fields);
- `asqa_eval_data_top5.provenance.json`: Provenance tracking hashes for each sample and its five documents;
- `asqa_eval_data_top5.manifest.json`: Fixed data source, revision, split, build rules, count, and file hashes.

Offline construction is also possible using the `--input-json` flag with a local raw Top-100 JSON file exported from the same fixed revision; the input must preserve the original `question`, `annotations`, and `docs` structures.



## Relationship with the Subsequent Frozen Set of 76 Items

This step produces only the 948 cleaned Top-5 input items. The subsequent established workflow is as follows:

```text
948 Top-5 input items
→ NO1: Retain 848
→ NO2: Retain 116
→ Hallucination injection: 116
→ NO3: Machine filtering retains 76
→ Manual review: 50 unchanged, 26 modified
→ Final frozen set: 76 items
```

Therefore, the difference in count between the 948 items and the final 76 items is both normal and necessary: ​​the former represents the complete candidate input prior to cleaning, while the latter constitutes the experimental frozen set resulting from machine filtering and manual review.