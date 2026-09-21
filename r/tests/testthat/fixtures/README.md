# Test fixtures

Author: John Grimes

Canned FHIR JSON responses used by the offline unit tests. Requests are routed
to these files by `helper-fixtures.R`; no network access is required.

The SNOMED CT content in `expand-single-page.json`, `expand-page-*.json` and
`closure-*.json` was captured from
`https://tx.ontoserver.csiro.au/fhir` on 2026-08-15 for the scope
`http://snomed.info/sct?fhir_vs=ecl/(>> 363346000)`, which is the example
documented in the repository README. The remaining fixtures are hand-written
variants that exercise specific edge cases.

## Expansion fixtures

| File | Purpose |
| ---- | ------- |
| `expand-single-page.json` | The whole scope in one page: 6 concepts, `total` 6, `offset` 0, with the R5 pre-adoption property extensions the server actually returns. Drives the documented 6x6 subsumption matrix and the 6x9 property widening. |
| `expand-page-1.json` | First page of the same scope with `count` 4: concepts 1-4, `total` 6, `offset` 0. |
| `expand-page-2.json` | Second page: concepts 5-6, `total` 6, `offset` 4. With `count` 4 and `total` 6 the paging loop makes exactly two requests. |
| `expand-missing-display.json` | Same 6 concepts, but `55342001` has no `display`, so its display must be recorded as `NA`. |
| `expand-empty.json` | `total` 0 and no `contains`; construction must fail with `Value set expansion is empty: <scope>`. |
| `expand-properties-preadopt.json` | 5 concepts carrying properties in the R5 pre-adoption extension format. |
| `expand-properties-native.json` | The same 5 concepts, same property semantics, expressed as native `contains.property` elements. Extraction from the two files must produce identical results. |

### Concepts in the captured scope

Expansion order defines both the row order and the first six columns of the
encoding.

| Position | Code | Display |
| -------- | ---- | ------- |
| 1 | 404684003 | Clinical finding |
| 2 | 64572001 | Disease |
| 3 | 363346000 | Malignant neoplastic disease |
| 4 | 399981008 | Neoplasm and/or hamartoma |
| 5 | 55342001 | Neoplastic disease |
| 6 | 138875005 | SNOMED CT Concept |

Associated morphology (`116676008`) is returned nested inside a role group
property (`609096000`), giving the three property features documented in the
README:

| Concept | Property feature |
| ------- | ---------------- |
| 363346000 | `609096000.116676008=1240414004` (Malignant neoplasm) |
| 399981008 | `609096000.116676008=400177003` (Neoplasm and/or hamartoma) |
| 55342001 | `609096000.116676008=108369006` (Neoplasm) |

The other three concepts carry no properties, so their property cells are 0.

### Property edge cases

`expand-properties-preadopt.json` and `expand-properties-native.json` cover, in
concept order:

1. `363346000` - one property with two subproperties
   (`609096000.116676008=1240414004`, `609096000.363698007=39937001`).
2. `399981008` - a subproperty nested inside a subproperty
   (`609096000.116676008=108369006`,
   `609096000.116676008.363698007=39937001`).
3. `55342001` - a flat property (`116676008=108369006`) alongside `parent` and
   `child` properties, which must be ignored.
4. `64572001` - a numeric property (`severityScore`, value 2.5), which becomes
   a single feature named `severityScore` carrying the number rather than an
   indicator column. `severityScore` is synthetic; SNOMED CT does not define
   it.
5. `404684003` - a malformed property with no `code` (skipped) plus a `parent`
   property (ignored), so this concept contributes no features at all.

The distinct property features are therefore, in the lexicographic (byte)
order that scikit-learn's `DictVectorizer` produces:

```text
116676008=108369006
609096000.116676008.363698007=39937001
609096000.116676008=108369006
609096000.116676008=1240414004
609096000.363698007=39937001
severityScore
```

Note that the outer role group property `609096000` has no value of its own, so
it contributes no feature - only its subproperties do.

## Closure fixtures

| File | Purpose |
| ---- | ------- |
| `closure-initialize.json` | Response to the `$closure` initialisation request: a `ConceptMap` with no `group`. |
| `closure-update.json` | Response to a single update covering all 6 concepts. Carries the 15 `subsumes` pairs the server actually returned, plus two deliberately non-`subsumes` targets that must be ignored. |
| `closure-update-batch-1.json` | New pairs from adding concepts 1-4 only (6 pairs), for use with the two-page expansion. |
| `closure-update-batch-2.json` | New pairs from then adding concepts 5-6 (9 pairs). Batches 1 and 2 together give the same 15 pairs as `closure-update.json`. |
| `closure-update-no-group.json` | An update response with no `group` at all, which must contribute zero pairs. |
| `closure-update-duplicates.json` | Reports the pair `(399981008, 404684003)` three times - twice in one group, once in a second - and reports `363346000` as subsuming itself, colliding with the identity cell. The encoding must stay multi-hot, so both cells must be exactly 1. `Matrix::sparseMatrix()` sums duplicate triplets, so this is what catches a missing deduplication step. |

### Pair orientation

A pair is `(element.code, target.code)` for every `group.element.target` whose
`equivalence` is `subsumes`. In FHIR R4 `subsumes` means "the source is-a the
target", so the target subsumes the element and the cell
`[row(element.code), col(target.code)]` is set to 1.

### Ignored equivalences in `closure-update.json`

Both are chosen so that honouring them would visibly corrupt the matrix:

- element `404684003`, target `55342001`, equivalence `equal` - would wrongly
  set `[404684003, 55342001]`.
- element `138875005`, target `363346000`, equivalence `specializes` - would
  wrongly set `[138875005, 363346000]`.

### Resulting matrix

Applying `closure-update.json` on top of the identity, over the six columns in
expansion order:

```text
             404684003 64572001 363346000 399981008 55342001 138875005
404684003            1        0         0         0        0         1
64572001             1        1         0         0        0         1
363346000            1        1         1         1        1         1
399981008            1        1         0         1        0         1
55342001             1        1         0         1        1         1
138875005            0        0         0         0        0         1
```

The rows for `399981008` and `363346000`, widened with the three property
columns, reproduce the output documented in the repository README:

```text
1 1 0 1 0 1 0 0 1
1 1 1 1 1 1 0 1 0
```
