# FHIR Terminology Encoder

Encoders that use a [FHIR](https://hl7.org/fhir/) terminology server to turn
terminology codes into features for machine learning.

You supply a scope in the form of a FHIR ValueSet URI and a FHIR terminology
endpoint. The encoder expands the scope, retrieves the subsumption
relationships between the concepts within it, and returns a multi-hot encoded
sparse matrix. Each code carries a feature for itself and for every concept
that subsumes it, so ontologically close codes share features. Concept
properties can be included as additional features.

This repository houses two implementations of the same idea:

| Language | Directory | Package |
| -------- | --------- | ------- |
| Python   | [`python/`](python) | `fhir-tx-encoder` (PyPI) |
| R        | [`r/`](r) | `fhirtxencoder` (GitHub) |

## Python

Install from PyPI:

```bash
pip install fhir-tx-encoder
```

```python
from fhir_tx import FhirTerminologyEncoder
import numpy as np

encoder = FhirTerminologyEncoder(
    # Ancestors of the SNOMED CT concept "Malignant neoplastic disease".
    scope="http://snomed.info/sct?fhir_vs=ecl/(%3E%3E%20363346000)",
    # Include "Associated morphology" (116676008) as a property.
    properties=["116676008"],
)

result = encoder.fit_transform(np.array([["399981008"], ["363346000"]]))
print(result.shape)  # (2, 9)
print(encoder.feature_names_)
```

The encoder implements the scikit-learn transformer interface, so it can be
used within a `Pipeline`. See [`python/README.md`](python/README.md) for the
full documentation.

## R

Install from GitHub:

```r
remotes::install_github("aehrc/fhir-tx-encoder", subdir = "r")
```

```r
library(fhirtxencoder)

encoder <- fhir_tx_encoder(
  # Ancestors of the SNOMED CT concept "Malignant neoplastic disease".
  scope = "http://snomed.info/sct?fhir_vs=ecl/(%3E%3E%20363346000)",
  # Include "Associated morphology" (116676008) as a property.
  properties = c("116676008")
)

result <- transform(encoder, matrix(c("399981008", "363346000"), ncol = 1))
dim(result)  # 2 9
encoder$feature_names
```

See [`r/README.md`](r/README.md) for the full documentation.

## Important note

This software is currently in alpha. It is not yet ready for production use.

Copyright © 2026, Commonwealth Scientific and Industrial Research Organisation
(CSIRO) ABN 41 687 119 230. Licensed under the
[Apache License, version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
