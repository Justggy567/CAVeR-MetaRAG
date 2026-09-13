# Canonical method specification

Status: The manuscript is the primary method authority. 

## Unit and probes

The routing unit is one source factoid. Each factoid has exactly five verified
statements:

1. original factoid;
2. synonym 1;
3. synonym 2;
4. antonym 1;
5. antonym 2.

The label space is YES, NO, and NOT_SURE. NOT_SURE has no deterministic reverse
label.

Preparation accepts a mutation side only when it contains exactly two non-empty
statements after stripping a leading bullet or numeric marker. A malformed synonym
or antonym side is regenerated with the identical frozen prompts, at most three format
attempts for that side. Rejected responses remain in the call ledger; no mutation is
manually completed. If all three attempts fail, the factoid is written to the
unresolved-mutation ledger and excluded from probe verification; preparation then
continues with the next factoid. The manifest reports the source, prepared, and
unresolved factoid counts and the unresolved rate. A response with no valid probe is
retained for audit but excluded from response-level verification and reported by ID.
These exclusions are missing probe-generation outcomes, not negative predictions.

Factoid decomposition is accepted only as a non-empty JSON array of non-empty
strings. Every response is logged before parsing. Invalid or truncated JSON is
retried with the identical frozen prompts for at most three attempts. Exhaustion is
recorded in a separate unresolved-decomposition ledger; the source response remains
in the prepared audit dataset with zero probes and is excluded from verification.
Its count, rate, and response ID are reported separately. No partial JSON is salvaged,
and the original answer is not silently substituted for a failed decomposition.

## Paper routing formula

For a deterministic original verdict, each synonym is anomalous when it differs
from the original verdict. Each antonym is anomalous when it differs from the
reverse of the original verdict.

An explicit paradox exists when the set of deterministic labels on the positive
side, original plus synonyms, intersects the set of deterministic labels on the
antonym side.

The structure-aware router escalates when at least one condition holds:

1. original is NOT_SURE;
2. explicit paradox;
3. both antonym probes are anomalous;
4. at least one synonym anomaly and at least one antonym anomaly occur.

Isolated synonym-only and isolated single non-paradox antonym anomalies do not
escalate. No anomaly-ratio shortcut is part of the paper method.

## Stage 2

The strong verifier re-verifies the identical original, synonyms, and antonyms.
Probes are not regenerated. The Stage-2 bundle replaces the Stage-1 bundle only
for routed factoids.

## Decision rule aligned with the manuscript

The implementation follows Section 3.2 of the manuscript. Positive-side penalties are YES=0, NOT_SURE=0.5, and NO=1; antonym-side penalties are NO=0, NOT_SURE=0.5, and YES=1. The factoid score is the mean of the five penalties. The response score is the maximum score over successfully prepared factoids, and a response is classified as hallucinated when its score is at least 0.5. Responses with no successfully prepared factoids are excluded from effectiveness evaluation and retained in the exclusion audit.
