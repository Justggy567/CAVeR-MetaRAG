# RQ3 Factoid Annotation Guide

Applicable to: Main-700, Replication-140, and subsequent independent reviews.

##  Annotation Unit and Displayed Information

Each unit consists of a `(question, context, factoid)` tuple. Annotation proceeds item by item according to the dataset's original order.

Strict blind review packages must display only:

- Anonymized `annotation_id` and `source_order`;
- `question`;
- `context`;
- `factoid`.

Do not display: previous category/support labels, generator or system predictions, labels from another annotator, candidate selection status, or label hints within filenames. Discrepancies are calculated and submitted to Annotator C for arbitration only after Annotators A and B have completed their work independently.

Priority of annotation decisions:

1. Determine whether the factoid constitutes a single, independently verifiable proposition;
2. Assign the primary reasoning category based solely on the factoid's semantics and structure;
3. Determine whether the factoid is explicitly supported based on the context;
4. Record evidence clarity;
5. Treat question relevance as a descriptive audit field; it must not override the first four criteria.

##  Atomicity

### PASS

The factoid expresses a core proposition that can be assigned a truth value independently. Permitted elements include:

- A subject accompanied by tight modifiers that constitute a single fact;
- Modifiers (time, location, units, or entities) necessary for disambiguation;
- Coordinated synonymous designations for the same relationship, provided no second independent assertion is introduced.

### FAIL

The factoid contains two or more propositions that could be true or false independently, or requires splitting for reliable annotation. Examples include:

- Connecting two independent facts using conjunctions like "and," "while," or "whereas";
- Simultaneously asserting multiple separable relationships (e.g., identity, timing, causality, comparison);
- A clause that functions as an additional proposition requiring verification rather than a necessary modifier;
- References or omissions that prevent the proposition from being interpreted independently within the current entry.

Operational test: If removing one component leaves a remainder that is still a complete fact capable of independent truth assessment, the item is generally non-atomic. Atomicity is based on propositional structure; do not judge mechanically based on sentence length or punctuation count. ## 3. Primary category

Based on the **primary reasoning operation** required to verify the factoid's truth value (not merely triggered by specific keywords); assign only one primary category.

| Label | Name | Core Question |
| --- | --- | --- |
| Cat1 | Entity / Alias ​​/ Lexical Relation | Does it primarily involve entities, aliases, categories, attributes, or direct semantic relations? |
| Cat2 | Negation / Exception | Does the truth value primarily depend on negation, exclusion, exceptions, permission/prohibition, or existence? |
| Cat3 | Quantifier / Boundary | Does the truth value primarily depend on scope, upper/lower limits, coverage, frequency, or quantifiers? |
| Cat4 | Time / Sequence | Does the truth value primarily depend on dates, duration, precedence, or the order of events? |
| Cat5 | Comparative / Superlative | Does it require comparing two or more objects, ranking, or identifying a maximum/minimum? |
| Cat6 | Numerical / Computation | Is the numerical value itself, or a calculation, count, ratio, or unit conversion, the core element? |
| Cat7 | Multi-entity / Multi-hop | Does it require combining two or more distinct facts/relations or reasoning along a chain of entities? |

### Conflict Resolution

- Numbers representing time or sequence: Cat4, not Cat6;
- Numbers acting as upper limits, lower limits, or coverage ranges: Cat3;
- Numbers requiring relative magnitude for assessment: Cat5;
- Multiple entities present but only a single direct relation needed: Do not classify as Cat7;
- Negation words present but not determining the primary truth value: Classify based on the actual primary operation;
- Multiple operations present: Consider whether verification is still possible if one operation is removed; retain the indispensable primary operation.

##  Context support

### TRUE / supported

The context explicitly states the factoid, or logically entails the factoid through brief, definitive reasoning that requires no external knowledge.
Allows for aliases, synonymous expressions, simple coreference resolution, and explicit logical combinations. ### FALSE / unsupported

The label FALSE is assigned in the following cases when using the binary RQ3 tag:

- The context explicitly contradicts the factoid;
- The context provides insufficient information to necessarily entail the factoid;
- The factoid holds true only by relying on external knowledge, common-sense speculation, or uncertain bridging inferences;
- The context supports an entity, time, scope, quantity, or relationship that is similar to but distinct from the one in the factoid;
- Evidence within the context is so conflicting or ambiguous that a definitive judgment of support cannot be made.

Core test: Based solely on the current context, could a cautious reader accept the factoid as necessarily true, rather than merely possibly true?
If yes, assign TRUE; otherwise, assign FALSE. Do not mistake the factoid's "potential correctness" in the real world for contextual support.

##  Evidence clarity

- `CLEAR`: The basis within the context for supporting or not supporting the factoid is sufficiently clear for the annotator to locate the decisive information;
- `AMBIGUOUS`: Text corruption, conflicting coreferences, contradictions, or unclear evidence boundaries make the support judgment unstable.

Evidence clarity differs from support: an entry where the evidence clearly indicates a lack of support or a clear contradiction can be labeled as
`context_support=FALSE, evidence_clarity=CLEAR`.

##  Question relevance

- `PASS`: The factoid directly or reasonably supports answering the question;
- `FAIL`: The factoid is irrelevant to the question, constitutes merely tangential information within the context, or does not help answer the question.

This field does not factor into the hard inclusion/exclusion criteria for the current Main-700 set, nor does it determine category or support status; it serves to describe the degree of
question-level alignment within the data. If future research adopts question relevance as an inclusion criterion, the dataset must be pre-registered and
reconstructed; entries cannot simply be silently deleted post-hoc. ## 7. Independent Annotation and Arbitration

1. Annotators A and B independently complete all items without seeing each other's results;
2. The taxonomy, field definitions, binary support criteria, and allowed values ​​are frozen prior to annotation;
3. Before calculating Inter-Annotator Agreement (IAA), item-by-item hashing is used to verify that the question, context, and factoid viewed by A and B are identical;
4. `atomicity`, `primary_category`, and `context_support` are designated as "hard fields"; any disagreement on these triggers arbitration by Annotator C;
5. `evidence_clarity` may trigger additional QC, whereas `question_relevance` defaults to merely reporting disagreements;
6. Annotator C reviews the source text and the labels from A and B to determine the final labels, without overwriting the original files from A and B;
7. The final JSONL file is reconstructed from consensus labels (where no disagreement existed) and C's arbitration labels, followed by item-by-item verification.


## Execution Guidelines for Replication-140

For Replication-140, Annotators A and B adhere to the same blind review protocols;