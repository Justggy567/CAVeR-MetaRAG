# Canonical method specification

Status: paper-aligned implementation baseline for revision. The manuscript is
the primary method authority. Any later change must update this file, Algorithm
1, Figure 1, tests, and the method-to-code map together.

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

## Decision rule currently made executable

The submitted manuscript names a fixed phi and MetaRAG aggregation rule but
does not give their complete numeric definitions. The canonical executable
baseline uses positive-side penalties YES=0, NOT_SURE=0.5, NO=1 and
antonym-side penalties NO=0, NOT_SURE=0.5, YES=1. The five penalties are
averaged per factoid; response score is the maximum factoid score; threshold is
0.5.

This paragraph is an explicit implementation assumption that must be confirmed
against the intended MetaRAG rule before paid reruns. If changed, it must be
changed once in scoring.py and all experiments rerun.
