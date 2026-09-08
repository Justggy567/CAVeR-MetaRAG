# Unified Post-processing Metrics for RQ2/RQ3

## Design Principles

During the model invocation phase, aggregate metrics for the paper are not calculated directly; instead, data is written to authoritative JSONL ledgers:

- `responses.jsonl`: Gold standard, Stage-1 predictions, final predictions, and question IDs;
- `factoids.jsonl`: Per-factoid Stage-1/Stage-2 scores, routing results, and trigger signals;
- `calls.jsonl`: Per-call tokens, latency, backend, model ID, and stage.

Metrics for RQ2/RQ3—covering escalation, rescue, harm, and efficiency—are reconstructed offline from the aforementioned ledgers after model execution completes. Consequently, adding statistical columns for the paper does not require re-invoking the models; re-running specific model experiments is necessary only if there are changes to factoid units, probes, model outputs, scoring rules, or routing rules.

## RQ2 Response-Level Metrics

Let `N` be the total number of responses, `R` the number of escalated responses, `F` the total number of factoids, `F_R` the number of escalated factoids, `C` the number of rescues, `H` the number of harms, and `E_R` the number of responses that were erroneous in Stage-1 and subsequently escalated.

| Output Field | Display Name in Paper | Formula |
|---|---|---|
| `response_escalation_rate` | Response Escalation Rate | `R / N` |
| `factoid_escalation_rate` | Factoid Escalation Rate | `F_R / F` |
| `rescue_rate_response` | Rescue Rate (Response) | `C / R` |
| `error_conditional_rescue_rate` | Error-conditional Rescue Rate | `C / E_R` |
| `harm_rate_response` | Harm Rate (Response) | `H / R` |

`rescue_rate_all_responses=C/N` and `harm_rate_all_responses=H/N` serve only as supplementary analyses. They do not represent the denominators used for Rescue/Harm in the original paper and cannot replace the paper's primary metrics.

## RQ3 Factoid-Level Metrics

In RQ3, each response corresponds to exactly one source factoid. Let `N` be the number of samples in a category and `R` be the number of escalations:

| Output Field | Display Name in Paper | Formula |
|---|---|---|
| `escalation_rate` | Escalation | `R / N` |
| `missed_detection_rate` | Missed Detection | `(Not escalated AND Stage-1 error) / N` |
| `over_escalation_rate` | Over-escalation | `(Escalated AND Stage-1 correct) / N` |
| `success_pass_rate` | Success Pass | `(Not escalated AND Stage-1 correct) / N` |
| `valid_escalation_rate` | Valid Escalation | `(Escalated AND Stage-1 error) / N` |
| `rescue_rate` | Rescue Rate | `Number of rescues / R` |
| `error_conditional_rescue_rate` | Error-cond. Rescue | `Number of rescues / Number of Valid Escalations` |
| `harm_rate` | Harm Rate | `Number of harms / R` |
| `escalation_efficiency_per_100` | Escalation Efficiency | `100 × (Rescues - Harms) / R` |

The analysis retains the integer numerator and denominator for each ratio. Ratios in the CSV are represented as decimals in the range `[0, 1]`; only `escalation_efficiency_per_100` uses the "net correct gain per 100 escalations" metric. Ratios are formatted as percentages during the paper's typesetting phase; the original decimal values ​​must not be overwritten in intermediate results.

## Consistency Constraints

Post-processing enforces the following validations:

1. `Missed + Over-escalation + Success Pass + Valid Escalation = N`;
2. `Escalation = Over-escalation + Valid Escalation`;
3. `Stage-1 errors = Missed + Valid Escalation`;
4. The final prediction for non-escalated responses cannot differ from the Stage-1 prediction;
5. `Net Correct Gain = Rescued - Harms`.

Failure to meet any constraint terminates the analysis to prevent the generation of paper tables that appear complete but contain internal inconsistencies. ## Zero Denominator

When there are no upgrade samples or no samples exhibiting "Stage-1 error and upgrade," the corresponding proportion is inestimable; the JSON records this as `null`, while the CSV leaves the field blank. It cannot be recorded as `0%` because "inestimable" differs in meaning from "observing a zero proportion."

## Output Pipeline

1. `analyze-policies`: Generates the RQ2 analysis JSON from the complete RQ2 dual-validator matrix;
2. `analyze-rq3`: Generates analysis JSON (both aggregate and category-specific) from the RQ3 primary policy ledger;
3. `export-publication`: Reads only the analysis JSON to generate:
- `rq2_policy_comparison.csv`; 
- `rq3_category_analysis.csv`.

The sole entry points for the analysis code are `src/caver/policy_analysis.py`, `src/caver/rq3_analysis.py`, and `src/caver/publication.py`. Historical patch scripts are not part of the public execution pipeline.