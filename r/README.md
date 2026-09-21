# FHIR Terminology Encoder for R

An R encoder that uses a [FHIR](https://hl7.org/fhir/) terminology server to turn
terminology codes into features for machine learning.

You supply a scope in the form of a FHIR ValueSet URI, and a FHIR terminology
endpoint. The encoder expands the scope, retrieves the subsumption relationships
between the concepts within it, and returns a multi-hot encoded sparse matrix.
Each code carries a feature for itself and for every concept that subsumes it,
so ontologically close codes share features. Concept properties can be included
as additional features.

A [recipes](https://recipes.tidymodels.org/) step is provided so the encoding can
participate in a tidymodels pipeline.

This is a port of the Python library in [`../python`](../python), and produces
the same output for the same inputs.

## Installation

```r
# install.packages("remotes")
remotes::install_github("aehrc/fhir-tx-encoder", subdir = "r")
```

The package is not on CRAN. It requires R 4.1 or later, and depends on httr2 and
Matrix. The recipes and tibble packages are only needed if you use
`step_fhir_tx()`.

## Usage

### Functional API

```r
library(fhirtxencoder)

encoder <- fhir_tx_encoder(
  # Ancestors of the SNOMED CT concept "Malignant neoplastic disease" (363346000)
  scope = "http://snomed.info/sct?fhir_vs=ecl/(%3E%3E%20363346000)",
  # Include "Associated morphology" (116676008) as a property
  properties = "116676008"
)

# Encode two SNOMED CT concepts:
# - "Neoplasm and/or hamartoma" (399981008)
# - "Malignant neoplastic disease" (363346000)
result <- transform(encoder, matrix(c("399981008", "363346000"), ncol = 1))

print(dim(result))
print(as.matrix(result))
print(encoder$feature_names)
```

Which outputs:

```
Expanding value set: http://snomed.info/sct?fhir_vs=ecl/(%3E%3E%20363346000)
Expanding (6 items, offset 0, total 6)
Expansion complete
Generating one-hot encoding... (6, 6)
Creating index... 6 items
Applying transitive closure...
Batch 1 of 1, 6 items... 15 pairs added
Subsumption encoding complete: (6, 6)
Encoding properties... (6, 9)
[1] 2 9
     [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
[1,]    1    1    0    1    0    1    0    0    1
[2,]    1    1    1    1    1    1    0    1    0
[1] "404684003"                      "64572001"                      
[3] "363346000"                      "399981008"                     
[5] "55342001"                       "138875005"                     
[7] "609096000.116676008=108369006"  "609096000.116676008=1240414004"
[9] "609096000.116676008=400177003" 
```

The first six features are the codes of the scope, in expansion order; the
remaining three are the property features. Row 1 encodes "Neoplasm and/or
hamartoma", which carries a 1 for itself (`399981008`), for the three concepts
in the scope that subsume it (`404684003`, `64572001`, `138875005`) and for its
associated morphology (`609096000.116676008=400177003`).

The result is a `Matrix::dgCMatrix`, so it stays sparse. Progress is reported
with `message()` and can be silenced with `suppressMessages()`.

Two further fields are useful for interpreting the columns:

```r
encoder$codes     # the codes, in column order
encoder$displays  # their display terms, NA where the server sent none
```

`encoder$displays` for the example above is:

```
[1] "Clinical finding"             "Disease"                     
[3] "Malignant neoplastic disease" "Neoplasm and/or hamartoma"   
[5] "Neoplastic disease"           "SNOMED CT Concept"           
```

Passing a matrix with more than one column encodes each column independently and
binds the encodings side by side, so a two-column input over a nine-feature
encoder gives eighteen columns.

Set `subsumption = FALSE` for a plain one-hot encoding, and `batch_size` to
control how many codes are sent to the server at a time (the default is 50,000).

### tidymodels

`step_fhir_tx()` replaces a column of codes with one numeric column per feature.

```r
library(recipes)
library(fhirtxencoder)

data <- data.frame(
  code = c("399981008", "363346000"),
  outcome = c(0, 1)
)

rec <- recipe(outcome ~ code, data = data) |>
  step_fhir_tx(
    code,
    scope = "http://snomed.info/sct?fhir_vs=ecl/(%3E%3E%20363346000)"
  ) |>
  prep()

bake(rec, new_data = data)
```

Which outputs (progress messages omitted):

```
# A tibble: 2 × 7
  outcome `404684003` `64572001` `363346000` `399981008` `55342001` `138875005`
    <dbl>       <dbl>      <dbl>       <dbl>       <dbl>      <dbl>       <dbl>
1       0           1          1           0           1          0           1
2       1           1          1           1           1          1           1
```

The encoder is fitted during `prep()` from `scope` alone. The terminology server
defines the feature space, so the training data plays no part in it and nothing
about the training data leaks into the encoding. `bake()` is then a lookup, and
works for any code within the scope whether or not it appeared in the training
data.

Because a recipe carries its data as a data frame, `bake()` produces dense
columns - one per feature, for every row. A scope of a few hundred concepts is
comfortable; a scope of tens of thousands is not, and is much better served by
calling `fhir_tx_encoder()` and `transform()` directly, which keeps the result
sparse.

### Errors

A code outside the scope, or an input that is not a two-dimensional character
matrix, is rejected:

```r
transform(encoder, matrix("404684002", ncol = 1))
#> Error: Encountered code not in scope: 404684002

transform(encoder, c("404684003"))
#> Error: X must be a two-dimensional character matrix
```

A scope that expands to nothing is rejected at construction rather than
producing a zero-width encoder:

```
Error: Value set expansion is empty: <scope>
```

## Terminology server support

The default endpoint is the CSIRO public Ontoserver,
`https://tx.ontoserver.csiro.au/fhir`. Pass `tx_url` to use another server.

Subsumption encoding relies on the `$closure` operation, which is optional in
the FHIR specification. A server that does not implement it can still be used
with `subsumption = FALSE`.

## Development

Run the unit tests, which answer every terminology server request from canned
fixtures in `tests/testthat/fixtures/`:

```bash
Rscript -e 'devtools::test("r")'
```

`tests/testthat/test-integration.R` is the exception: it runs the example above
against the live CSIRO public server. It is skipped when `NOT_CRAN` is unset (so
`R CMD check` and covr never reach it), when the machine is offline, and when
the `CI` environment variable is set. A terminology server outage therefore
cannot break a scheduled check. `devtools::test()` sets `NOT_CRAN`, and a local
shell leaves `CI` unset, so the live test runs by default when you run the suite
yourself.

Run the package check and coverage:

```bash
R CMD build r && R CMD check --as-cran fhirtxencoder_1.0.0.tar.gz
Rscript -e 'covr::package_coverage("r")'
```

## Important note

This software is currently in alpha. It is not yet ready for production use.

Copyright © 2026, Commonwealth Scientific and Industrial Research Organisation
(CSIRO) ABN 41 687 119 230. Licensed under the
[Apache License, version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
