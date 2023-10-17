# FHIR terminology encoder

This is a scikit-learn compatible encoder that uses a FHIR terminology server to
encode ontological features.

Currently it supports subsumption relationships.

You supply a scope in the form of a FHIR ValueSet URI, and a FHIR terminology
endpoint.

The result is a multi-hot encoded vector delivered as a sparse matrix, suitable
for input into most models and estimators.

## Installation

```bash
pip install 'fhir-tx-encoder @ git+https://github.com/aehrc/fhir-tx-encoder@main'
```

## Usage

```python
from fhir_tx_encoder import FhirTerminologyEncoder
import numpy as np

encoder = FhirTerminologyEncoder(
    scope="http://snomed.info/sct?fhir_vs=ecl/(%3C%3C%20404684003)",
    tx_url="http://localhost:8081/fhir",
)

result = encoder.fit_transform(np.array([["399981008", "363346000"]]))
print(f"result.shape: {result.shape}")
print(f"result:\n{result.shape}")
```

Which would output:

```
Expanding value set: http://snomed.info/sct?fhir_vs=ecl/(%3E%3E%20363346000)
Expanding (6 items, offset 0, total 6)
Expansion complete
Generating one-hot encoding... (6, 6)
Creating index... 6 items
Applying transitive closure
Batch 1 of 1, 6 items... 15 pairs added
Encoding complete: (6, 6)
encoder.codes_: ['404684003', '64572001', '363346000', '399981008', '55342001', '138875005']
encoder.displays_: ['Clinical finding', 'Disease', 'Malignant neoplastic disease', 'Neoplasm and/or hamartoma', 'Neoplastic disease', 'SNOMED CT Concept']
result.shape: (2, 6)
result:
[[1. 1. 0. 1. 0. 1.]
 [1. 1. 1. 1. 1. 1.]]
```

Copyright © 2023, Commonwealth Scientific and Industrial Research Organisation 
(CSIRO) ABN 41 687 119 230.
