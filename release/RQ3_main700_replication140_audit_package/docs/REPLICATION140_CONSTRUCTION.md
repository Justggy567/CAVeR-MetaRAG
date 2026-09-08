# Replication-140 Independent Replication Subset Construction Process

## Objective

Replication-140 is designed to verify whether the RQ3 results hold true on a new, structurally balanced diagnostic sample, without reusing data from the Main-700 project. It is not a hold-out set sampled from Main-700; instead, it is constructed by first excluding entries that overlap with Main-700 from a versioned candidate pool, followed by deterministic selection based on pre-fixed strata.

## Input and Filtering

The input consists of 830 entries from `data/construction/rq3_candidate_pool_final.jsonl`. Each candidate is compared against Main-700 using the following four keys:

1. `normalize(source_dataset) + normalize(source_id)`;
2. Normalized question;
3. Normalized context;
4. Normalized factoid.

Text normalization involves Unicode NFKC, case folding, replacing non-alphanumeric characters with spaces, and collapsing consecutive whitespace. Any match on any of these keys against Main-700 results in exclusion.

479 entries were excluded, leaving 351 entries.

The filtering results are saved in `data/construction/rq3_replication_candidate_pool_351.jsonl`.

## Stratified Deterministic Sampling

The 351 entries are divided into 28 strata based on the following three-dimensional cross-classification:

```text
7 categories × 2 context-support states × 2 sources = 28 strata
```

The sorting key for each candidate is:

```text
SHA256("rq3-replication140-v1|" + candidate_id)
```

Following a fixed traversal order—Cat1 through Cat7, support=True through False, and ASQA through HALUEVALQA—entries within each stratum are sorted in ascending order by hash and candidate ID. The top 5 entries are selected, subject to a global uniqueness constraint on the normalized context. This yields:

- 28 × 5 = 140 items;
- 20 items per category;
- supported / unsupported = 70 / 70;
- ASQA / HALUEVALQA = 70 / 70;
- 211 unselected records moved to `rq3_replication140_reserve_queue.jsonl`.


##  Independence and Internal Duplication Checks

The overlap count between the final 140 items and Main-700 is:

| Check Key | Overlap Count |
| --- | ---: |
| source_dataset + source_id | 0 |
| normalized question | 0 |
| normalized context | 0 |
| normalized factoid | 0 |

IDs within Replication-140 are unique; furthermore, the selection order in the final dataset, candidate ID, stratum, and
sample priority can all be deterministically reconstructed from the script.


##  Annotation and Adjudication

Files A and B cover the same 140 IDs. Prior to adjudication, category agreement was 132/140 (94.29%, κ=0.9335),
and support agreement was 140/140 (100%, κ=1.0000). There were 8 unique hard-field
disagreements (regarding atomicity, category, or support), all occurring within the 10 records assigned to C;
an additional 2 cases involved clarity-related quality control.

Post-adjudication, the net change in category/support relative to the deterministic sampling file was zero,
and all final atomicity statuses were PASS.
This indicates that C's review upheld the current canonical labels, not that there were no initial disagreements between A and B.
The updated A/B audit table aligns 100% (140/140) with the frozen set regarding questions, contexts, and factoids.

##  Replication Commands and Inputs

To run the full package-level verification:

```powershell
python scripts/verify_package.py
```