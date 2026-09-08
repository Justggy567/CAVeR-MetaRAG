# RQ3 taxonomy v2

Original file SHA-256:
`0f38475b49193009c1dcae932b893fcf7d153a275778089a5420f58f665edaf7`

General Principle: Each factoid is assigned to a primary category based on the **main reasoning operation** required to verify its truth against the given context, rather than based on superficial lexical cues.

1. **Entity / Alias ​​/ Lexical Relation**: Relies primarily on entities, aliases, categories, membership, attributes, or direct lexical/semantic relations. Purely synonymous surface variations do not constitute a factual difference.
2. **Negation / Exception**: Relies primarily on propositional polarity, exclusion, exceptions, permission/prohibition, existence/non-existence, or explicit negation/exception relations.
3. **Quantifier / Boundary**: Relies primarily on scope, upper/lower limits, coverage, frequency, or quantification/boundary relations such as *all*, *some*, *only*, *at least*, or *at most*.
4. **Time / Sequence**: Relies primarily on dates, years, durations, temporal positioning, event ordering, or before/after relationships.
5. **Comparative / Superlative**: Relies primarily on comparisons, rankings, relative magnitude, or superlative relations such as *highest/lowest* or *maximum/minimum*.
6. **Numerical / Computation**: Assigned here only if the numerical value, quantity, measurement, ratio, unit, or calculation result itself is the primary fact to be verified, or if counting, arithmetic, ratios, or unit conversions are explicitly required. The mere presence of numbers is insufficient; temporal numbers take precedence under **Time / Sequence**, boundary quantities under **Quantifier / Boundary**, and numbers determining truth via comparison under **Comparative / Superlative**.
7. **Multi-entity / Multi-hop**: Requires combining multiple entity relations, two or more distinct contextual facts, or verification along a multi-step relational chain. The mere presence of multiple entity names is insufficient; two or more evidentiary relations or facts must be combined. The code tags `Cat1`–`Cat7` are retained for data traceability; the display names used in the paper and the aforementioned definitions are fixed in `src/caver/taxonomy.py`. Name mapping does not replace independent manual annotation, conflict resolution, or IAA.