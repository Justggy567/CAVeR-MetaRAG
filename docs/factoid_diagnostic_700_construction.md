# Construction and Final Audit of the RQ3 Main-700 Diagnostic Set

## 1. Purpose and unit of analysis

RQ3 uses a factoid-level diagnostic set to examine how the fixed structure-aware cascade behaves across different reasoning requirements. The experimental unit is one source factoid. RQ3 therefore does not perform answer decomposition at run time.

The frozen Main-700 set contains seven mutually exclusive reasoning categories crossed with two binary context-support states:

```text
7 categories × (50 supported + 50 unsupported) = 700 factoids
```

The set is diagnostic and deliberately balanced. It is not intended to estimate the natural prevalence of the seven reasoning categories in ASQA or HaluEvalQA.

## 2. Revised taxonomy

Every factoid is assigned one primary category according to the principal reasoning operation required to verify the claim against the supplied context, rather than according to surface keywords.

1. **Entity / Alias / Lexical Relation (Cat1).** Verification primarily requires identifying an entity, alias, category, membership relation, attribute, or direct lexical/semantic relation.
2. **Negation / Exception (Cat2).** Truth primarily depends on polarity, exclusion, exception, permission/prohibition, or existence/non-existence.
3. **Quantifier / Boundary (Cat3).** Verification primarily depends on scope, lower or upper bounds, coverage, frequency, or quantification/boundary relations.
4. **Time / Sequence (Cat4).** Verification primarily depends on dates, duration, temporal position, event order, or before/after relations.
5. **Comparative / Superlative (Cat5).** Verification primarily requires a comparative, ranking, relative-magnitude, or superlative relation.
6. **Numerical / Computation (Cat6).** A numerical value, quantity, measurement, proportion, unit, or explicit computation is the principal factual element. The mere presence of a number is insufficient.
7. **Multi-entity / Multi-hop (Cat7).** Verification requires combining two or more evidence relations, contextual facts, or entity relations.

Temporal numbers take Cat4 priority, boundary quantities take Cat3 priority, and comparison-dependent numbers take Cat5 priority. No keyword or regular-expression classifier supplies the final category.

## 3. Legacy-pool review

The original 700-record diagnostic set was treated as a legacy audit pool. Two annotators independently reviewed the question, context, and factoid under the revised decision guide. The review did not use router outcomes, verifier identity, escalation benefit, or other experimental outputs.

For the legacy pool, raw agreement was 81.6% for the primary category and 92.1% for binary context support. The corresponding Cohen's kappa values were 0.732 and 0.845. The annotators disagreed on at least one core field for 222 records, which were reviewed by a third adjudicator.

After adjudication:

- 206 legacy records were retained without label changes;
- 126 were retained with revised labels;
- 368 were excluded from the eligible legacy pool.

The final Main-700 did not transfer every eligible legacy record automatically. Source-aware selection was used to reduce excessive concentration from either ASQA or HaluEvalQA while preserving the required Category × Support composition. A total of 263 legacy factoids were retained in Main-700.

## 4. Candidate-pool construction

The replenishment pool was constructed from the native ASQA and HaluEvalQA source collections. Candidate generation was guided by deficits in the fourteen Category × Support strata.

Supported candidates were constructed from question-answer information and checked against the supplied context. Unsupported candidates were complete claims for which the supplied context did not provide direct support. The requested category during candidate generation was only a sampling target and was never treated as the final human label.

Candidate records were reviewed before entering the eligible pool. The three hard eligibility decisions used for Main-700 construction were:

1. the factoid expresses one atomic, independently verifiable proposition;
2. the final primary category conforms to the revised taxonomy; and
3. the binary context-support label correctly represents whether the supplied context directly supports the factoid.

Question relevance and evidence clarity were recorded as descriptive diagnostic fields where available. **Question relevance was not used as a hard dataset exclusion or replacement criterion for the final Main-700 set.** It must therefore not be described in the manuscript or response letter as a mandatory screening filter.

From the eligible candidate pool, 437 records were selected for Main-700. Combined with 263 retained legacy records, this yielded the final 700 factoids.

## 5. Final assembly and freezing

The frozen set contains 100 records in each category, with 50 context-supported and 50 context-unsupported factoids per category. It contains 229 source questions/contexts from ASQA and 471 from HaluEvalQA.

Each record contains:

- a stable `MAIN_xxxx` identifier;
- source dataset and source identifier;
- question;
- context;
- source factoid;
- final primary category; and
- final binary context-support label.

The canonical frozen file is `data/frozen/rq3_main700_final.jsonl`. Its SHA-256 for the final audited version is:

```text
c3f868a3af409dbf61838f4edce8820c01a8a62f7c7fe70c3457a94dd5db331a
```

Any content or label change creates a new dataset version and invalidates prepared probes, caches, run ledgers, and downstream analysis derived from the previous hash.

## 6. Final Main-700 double review

After final assembly, two annotators independently reviewed all 700 frozen records. The core final-audit questions were:

1. Does the factoid express one atomic, independently verifiable proposition?
2. Does the assigned primary category follow the revised taxonomy?
3. Does the binary context-support label correctly indicate direct support by the supplied context?

The annotation tables also retained question relevance and evidence clarity for descriptive auditing. These auxiliary fields did not override the three hard final-audit criteria.

### 6.1 Pre-adjudication agreement

| Field | Agreement | Cohen's kappa | Interpretation |
| --- | ---: | ---: | --- |
| Primary category | 672/700 = 96.00% | 0.9533 | very high agreement |
| Context support | 678/700 = 96.86% | 0.9371 | very high agreement |
| Question relevance (descriptive only) | 700/700 = 100% | 1.0000 | complete agreement; 97 FAIL and 603 PASS labels per annotator |
| Atomicity (hard criterion) | 700/700 = 100% | not estimable | both annotators marked all records PASS; zero label variance |
| Evidence clarity (descriptive only) | 700/700 = 100% | not estimable | zero label variance |

Cohen's kappa is not reported as 1.0 for atomicity or evidence clarity because both annotators assigned only one label, making chance-corrected agreement mathematically non-estimable.

### 6.2 Adjudication and final-label consistency

The adjudication file contains 53 non-empty review rows. Forty-seven records contained a category and/or context-support disagreement. All disagreements involving the three hard validity fields were adjudicated before freezing. The remaining auxiliary review rows are retained for traceability and are not counted as additional hard-field disagreements.

A record-level consistency check found:

- 700/700 atomicity PASS decisions from both annotators, with no atomicity disagreement;
- 0 final category mismatches between the frozen JSONL and adjudicated labels;
- 0 final context-support mismatches between the frozen JSONL and adjudicated labels; and
- 0 unadjudicated disagreements involving the three hard validity fields.

The final audit supports the atomicity, category, and context-support validity criteria used by the RQ3 experiment. Question relevance was not a mandatory inclusion criterion.

## 7. Audit provenance

The final review evidence consists of:

- `RQ3_factoid700_annotator_A.csv`;
- `RQ3_factoid700_annotator_B.csv`; and
- `RQ3_factoid700_adjudication.csv`.

The file hashes recorded on 2026-08-29 are:

| File | SHA-256 |
| --- | --- |
| Annotator A | `86118492234400d91c79035769fe8f8f2f103a942c9d1888042f53448f6c22e3` |
| Annotator B | `d9ccbace57835264b4bbb9bf85a50557981533b3feb59d9de1bb5f1d77dc936` |
| Adjudication | `c34dc6b19475b09c513be2afcd9f092a016df8ebb6ff88ebf7a66450b1a24c6a` |

The original annotation files, pre-adjudication agreement report, adjudication table, final frozen data hash, and analysis output must be archived together. Experimental results must be generated only from the final frozen JSONL labels; annotation CSV files are audit evidence and are not run-time label override files.

## 8. Limits of the construction

Main-700 is a manually curated, balanced diagnostic set. Its category frequencies are fixed by design and should not be interpreted as population prevalence. The same source corpora and written taxonomy were used throughout construction, so replication on a disjoint factoid set is reported separately. Human review reduces label noise but does not remove all judgment dependence, especially for claims that plausibly require more than one reasoning operation.
