# Method-to-code map

| Manuscript element | Canonical code | Test evidence |
|---|---|---|
| Three-way labels | src/caver/labels.py | tests/test_labels.py |
| Original plus two synonyms and two antonyms | src/caver/schema.py, probes.py | tests/test_cascade.py |
| Original uncertainty | src/caver/routing.py | tests/test_routing.py |
| Relative synonym/antonym anomalies | src/caver/routing.py | tests/test_routing.py |
| Explicit paradox | src/caver/routing.py | tests/test_routing.py |
| Four-condition router | src/caver/routing.py | 243-combination exhaustive test |
| Stage-2 reuse of the same probes | src/caver/cascade.py | tests/test_cascade.py |
| Factoid score and response aggregation | src/caver/scoring.py | tests/test_scoring.py |
| Response confusion matrix and unified rescue/harm denominators | src/caver/metrics.py, policy_analysis.py | tests/test_metrics.py, tests/test_policy_analysis.py |
| RQ3 nine paper metrics and routing partition invariants | src/caver/rq3_analysis.py | tests/test_rq3_analysis.py |
| Token, call, and latency separation | src/caver/schema.py, ledger.py | tests/test_ledger.py |
| Frozen dataset/probes | src/caver/datasets.py, generation.py | preparation manifest |
| Decomposition JSON validation, retry, unresolved-response ledger, and exclusion audit | src/caver/generation.py, experiment.py | tests/test_generation.py, tests/test_experiment.py |
| Exact-two mutation validation, format retry, unresolved ledger, and coverage audit | src/caver/generation.py, experiment.py | tests/test_generation.py, tests/test_experiment.py |
| Matched verifier-output cache | src/caver/cache.py | tests/test_cache.py |
| Public execution path | main.py, src/caver/cli.py | tests/test_cli.py |
| Reviewer parameter completeness | src/caver/configuration.py, audit.py | audit-parameters, tests/test_revision_protocol.py |
| Cold/warm and resource protocol | configs/reproducibility_protocol.json, experiment.py | warmup_calls.jsonl + run manifest |
| Complete RQ2/RQ3 manuscript CSV generation | src/caver/publication.py | tests/test_publication.py |

Legacy files metaragT_2.py, metarag03.py, RQ1_code.py,
RQ2_gemma3_deepseek.py, RQ3_3types.py, and
RQ3_structure_aware_data_statistics.py are not imported by the public entry
point. They remain unchanged as historical artifacts until provenance review.
