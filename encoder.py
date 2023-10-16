import numpy as np
import numpy.typing as npt
import requests
from scipy.sparse import csr_matrix, hstack, lil_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from closure import Closure


class FhirTerminologyEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes terminology codes into a vector representations that contain information about their
    relationships to other codes within the terminology.
    """

    def __init__(
        self,
        scope: str,
        tx_url: str = "https://tx.ontoserver.csiro.au/fhir",
        batch_size: int = 50000,
    ):
        print(f"Expanding value set: {scope}")
        coding_batches = self._expand_scope(scope, tx_url, batch_size)
        print("Expansion complete")

        print("Generating one-hot encoding...", end=" ")
        ohe = OneHotEncoder(categories=[self.categories_])
        # Use a lil_matrix as an intermediate representation that enables efficient updates.
        self._encoded = lil_matrix(
            ohe.fit_transform(np.array(self.categories_).reshape(-1, 1))
        )
        print(self._encoded.shape)

        print("Creating index...", end=" ")
        self._index = {value: index for index, value in enumerate(ohe.categories_[0])}
        print(f"{len(self._index)} items")

        print("Applying transitive closure")
        self._apply_closure(coding_batches, tx_url)
        print("Closure complete")

        # Convert the final product back to a csr_matrix for efficient arithmetic operations.
        self._encoded = self._encoded.tocsr()

    def _expand_scope(self, scope, tx_url, batch_size):
        offset = 0
        total = 0
        self.categories_ = []
        coding_batches = []
        while offset <= total:
            response = requests.get(
                f"{tx_url}/ValueSet/$expand",
                params={"url": scope, "count": batch_size, "offset": offset},
            )
            response.raise_for_status()
            response_json = response.json()

            print(
                f"Expanding ({len(response_json['expansion']['contains'])} items, offset {offset})"
            )
            codings = response_json["expansion"]["contains"]
            coding_batches.append(codings)
            self.categories_.extend([coding["code"] for coding in codings])

            total = response_json["expansion"]["total"]
            offset += batch_size
        return coding_batches

    def _apply_closure(self, coding_batches, tx_url):
        closure = Closure(tx_url=tx_url)
        num_batches = len(coding_batches)
        for i, batch in enumerate(coding_batches):
            print(f"Batch {i + 1} of {num_batches}, {len(batch)} items...", end=" ")
            pairs = closure.update(batch)
            for pair in pairs:
                x = self._index[pair[0]]
                y = self._index[pair[1]]
                self._encoded[x, y] = 1
            print(f"{len(pairs)} pairs added")

    def transform_column(self, X: npt.NDArray) -> csr_matrix:
        try:
            X_indices = np.array([self._index[x] for x in X])
        except KeyError as e:
            raise ValueError(f"Encountered code not in scope: {e}")
        return self._encoded[X_indices]

    def transform(self, X: npt.NDArray[npt.NDArray]) -> csr_matrix:
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a Numpy array")
        if len(X.shape) != 2:
            raise ValueError("X must be a two-dimensional array")

        stacked = hstack([self.transform_column(x) for x in X])
        return stacked

    def fit(self, X, y=None):
        return self
