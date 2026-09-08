# ASQA / HaluEvalQA Data Cleaning Reproduction Package

This package restores the original data cleaning sequence: like ASQA, HaluEvalQA proceeds directly from the raw data to the NO1 stage.

## ASQA 948 Source Records → Top-5 Input

For the construction steps leading up to the NO1 stage for ASQA, refer to [provenance/asqa/ASQA_948_TO_TOP5_CONSTRUCTION_CN.md](provenance/asqa/ASQA_948_TO_TOP5_CONSTRUCTION_CN.md); executable code is available at [provenance/asqa/build_asqa_top5.py](provenance/asqa/build_asqa_top5.py). This step reads 948 records from the fixed revision of the `Self-GRIT/asqa_eval` full `valid` split and selects the top five documents for each record based on the original retrieval order—without sampling, re-retrieval, or re-ranking.

## Fixed Pipeline

ASQA:

```text
Raw ASQA
→ NO1: 40% core keyword coverage rule
→ NO2: Context support check for the original correct answer
→ Hallucination injection: User-specified new INJECT_SYSTEM_PROMPT
→ NO3: Original double-check
① right_answer is fully supported by context
② hallucinated_answer indeed contains a hallucination
```

HaluEvalQA:

```text
Raw HaluEvalQA
→ NO1: 40% core keyword coverage rule
→ NO2: Context support check for the original correct answer
→ NO3: Original double-check
① right_answer is fully supported by context
② hallucinated_answer indeed contains a hallucination
```

HaluEvalQA uses the original `right_answer` and `hallucinated_answer`; no additional answer rewriting or hallucination injection is performed.

## Running in PyCharm

Edit [simple_cleaning/config.py](simple_cleaning/config.py), then run [run_cleaning.py](run_cleaning.py) directly. ```python
DATASETS_TO_RUN = ["asqa"]              # Or ["haluevalqa"], or run both
TEST_LIMIT = 10                          # Set to None for a full-scale production run
RUN_TAG = "your_new_run_tag"
DRY_RUN = False
RESUME = True
```

The API key is read exclusively from the `DEEPSEEK_API_KEY` environment variable. The model is fixed to `deepseek-v4-flash` with thinking `disabled`. `temperature=0` and `max_tokens=5` are used for NO2 and NO3; `temperature=0.7` is used for hallucination injection in ASQA.

## Data Location

By default, data is read from the `data` directory located one level above this package:

```text
data/
├── asqa_eval_data.json
└── halueval_qa_data.json
```

Alternatively, the data directory can be specified using the `METARAG_DATA_DIR` environment variable. NO1 Standardized output format:

```json
{
"id": "...",
"question": "...",
"context": "...",
"right_answer": "...",
"hallucinated_answer": "..."
}
```

## Output Files

Output location:

```text
runs_simple_v4/<dataset>_<RUN_TAG>/
```

Key results:

```text
00_normalized.json
01_no1_clean.json
01_no1_rejected.json
02_no2_clean.json
02_no2_rejected.json
02_no2_unresolved.json
03_injected.json                       # ASQA only
03_injection_unresolved.json           # ASQA only
04_no3_clean.json
04_no3_rejected.json
04_no3_unresolved.json
final_clean.json
run_manifest.json
protocol_snapshot.json
ledgers/no1.json
ledgers/no2.jsonl
ledgers/injection.jsonl                # ASQA only
ledgers/no3_right.jsonl
ledgers/no3_hallucination.jsonl
```

## Testing and Experimental Scope

Offline tests do not call real APIs:

```powershell
python -m unittest discover -s tests -v
```

Passing unit tests only confirms the correctness of code paths, prompt locking, and ledger logic; it does not indicate that the actual data cleaning experiment has been executed. The data construction process is considered complete only when `final_clean.json` has been generated, the manifest status is `complete`, and the count of unresolved items at each stage is zero.