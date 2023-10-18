#
#     Copyright © 2023 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
#     ABN 41 687 119 230.
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

import numpy as np
import numpy.typing as npt
import requests
from scipy.sparse import csr_matrix, hstack, lil_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder

from .closure import Closure

EXPANSION_PROPERTY_PREADOPT = (
    "http://hl7.org/fhir/5.0/StructureDefinition/"
    "extension-ValueSet.expansion.contains.property"
)

IGNORED_PROPERTIES = ["parent", "child"]


class FhirTerminologyEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes terminology codes into a vector representations that contain information about their
    relationships to other codes within the terminology.

    Currently this is limited to subsumption relationships as reported by the closure operation of
    the terminology server.

    :ivar codes_: The list of codes that are included in the scope, in the order that they are
        represented within the columns of the encoding.
    :type codes_: list[str]
    :ivar displays_: The list of display names for the codes that are included in the scope, in the
        order that they are represented within the columns of the encoding.
    :type displays_: list[str]
    """

    def __init__(
        self,
        scope: str,
        tx_url: str = "https://tx.ontoserver.csiro.au/fhir",
        properties: list[str] = None,
        batch_size: int = 50000,
    ):
        """
        :param scope: A FHIR ValueSet URI that defines the scope of the codes to be encoded.
        :param tx_url: A FHIR terminology server endpoint.
        :param properties: A list of properties to include in the encoding. A single value of "*"
            will include all properties.
        :param batch_size: The number of codes to send to the terminology server at a time when
            running queries.
        """
        print(f"Expanding value set: {scope}")
        coding_batches = self._expand_scope(scope, tx_url, properties, batch_size)
        print("Expansion complete")

        print("Generating one-hot encoding...", end=" ")
        ohe = OneHotEncoder(categories=[self.codes_])
        # Use a lil_matrix as an intermediate representation that enables efficient updates.
        self._encoded = lil_matrix(
            ohe.fit_transform(np.array(self.codes_).reshape(-1, 1))
        )
        print(self._encoded.shape)

        # Create an index of code -> index, which we use to update the correct row and column in the
        # matrix when making subsumption updates.
        print("Creating index...", end=" ")
        self._index = {value: index for index, value in enumerate(ohe.categories_[0])}
        print(f"{len(self._index)} items")

        print("Applying transitive closure...")
        self._apply_closure(coding_batches, tx_url)
        print(f"Subsumption encoding complete: {self._encoded.shape}")

        print("Encoding attributes...", end=" ")
        dv = DictVectorizer()
        encoded_attributes = dv.fit_transform(self.properties_)
        self._encoded = hstack([self._encoded, encoded_attributes])
        print(self._encoded.shape)

        # Create a list of feature names that includes the codes and the attributes.
        self.feature_names_ = self.codes_ + dv.feature_names_

        # Convert the final product back to a csr_matrix for efficient arithmetic operations.
        self._encoded = self._encoded.tocsr()

    def _expand_scope(self, scope, tx_url, properties, batch_size):
        """
        Get the list of all codes in the scope.
        """
        offset = 0
        total = 0
        self.codes_ = []
        self.displays_ = []
        self.properties_ = []
        coding_batches = []

        # Run an expand request with a count equal to the batch size, and iterate until we have
        # retrieved all codes.
        while offset <= total:
            params = {
                "url": scope,
                "count": batch_size,
                "offset": offset,
            }
            params = [
                ("url", scope),
                ("count", batch_size),
                ("offset", offset),
            ]
            if properties is not None:
                for p in properties:
                    params.append(("property", p))
            response = requests.get(
                f"{tx_url}/ValueSet/$expand",
                params=params,
            )
            response.raise_for_status()
            response_json = response.json()
            total = response_json["expansion"]["total"]

            print(
                f"Expanding ({len(response_json['expansion']['contains'])} items, offset {offset}, "
                f"total {total})"
            )
            codings = response_json["expansion"]["contains"]

            # Add the batch of codings to the list of batches. These same batches will be used later
            # in the closure operation.
            coding_batches.append(codings)

            # Add each code and display to a list. These are useful for later retrieval for the
            # creation of feature name dictionaries.
            self.codes_.extend([coding["code"] for coding in codings])
            self.displays_.extend(
                [
                    (coding["display"] if "display" in coding else None)
                    for coding in codings
                ]
            )
            if properties is not None:
                self.properties_.extend(
                    [properties_to_dict(coding) for coding in codings]
                )

            # The offset of the next query is the current offset plus the batch size.
            offset += batch_size

        return coding_batches

    def _apply_closure(self, coding_batches, tx_url):
        """
        Perform a closure operation on all the codes in the scope and update the encoded matrix with
        the subsumption relationships.
        """
        closure = Closure(tx_url=tx_url)
        num_batches = len(coding_batches)

        for i, batch in enumerate(coding_batches):
            print(f"Batch {i + 1} of {num_batches}, {len(batch)} items...", end=" ")
            # Get the new pairs that result from adding a batch of codes to the closure table.
            pairs = closure.update(batch)
            for pair in pairs:
                # Use the index to find the correct x and y coordinates in the matrix.
                x = self._index[pair[0]]
                y = self._index[pair[1]]
                # Update the matrix with the new subsumption relationship.
                self._encoded[x, y] = 1
            print(f"{len(pairs)} pairs added")

    def transform_column(self, X: npt.NDArray) -> csr_matrix:
        """
        Retrieve the encodings for a single column of codes.
        """
        try:
            X_indices = np.array([self._index[x] for x in X])
        except KeyError as e:
            raise ValueError(f"Encountered code not in scope: {e}")
        return self._encoded[X_indices]

    def transform(self, X: npt.NDArray[npt.NDArray]) -> csr_matrix:
        """
        Retrieve the encodings for a two-dimensional array of codes.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a Numpy array")
        if len(X.shape) != 2:
            raise ValueError("X must be a two-dimensional array")

        stacked = hstack([self.transform_column(x) for x in X])
        return stacked

    def fit(self, X, y=None):
        """
        Does nothing, but required for compatibility with scikit-learn pipelines.
        """
        return self


def properties_to_dict(coding: dict):
    try:
        result = {}
        if "property" in coding:
            for property_element in coding["property"]:
                code = property_element["code"]
                if code not in IGNORED_PROPERTIES:
                    value_key = next(k for k in coding.keys() if k.startswith("value"))
                    result[code] = property_element[value_key]
                    add_subproperties_to_dict(property_element, code, result)
        elif "extension" in coding:
            property_extensions = [
                e
                for e in coding["extension"]
                if e["url"] == EXPANSION_PROPERTY_PREADOPT
            ]
            for property_extension in property_extensions:
                code_extension = next(
                    e for e in property_extension["extension"] if e["url"] == "code"
                )
                code = code_extension["valueCode"]
                if code not in IGNORED_PROPERTIES:
                    value_extension = next(
                        (
                            e
                            for e in property_extension["extension"]
                            if e["url"] == "value"
                        ),
                        None,
                    )
                    if value_extension is not None:
                        value_key = next(
                            k for k in value_extension.keys() if k.startswith("value")
                        )
                        result[code] = value_extension[value_key]
                    add_subproperties_to_dict(property_extension, code, result)

    except StopIteration:
        return {}
    return result


def add_subproperties_to_dict(property_element: dict, code: str, result: dict):
    if "extension" in property_element:
        subproperty_extensions = [
            e for e in property_element["extension"] if e["url"] == "subproperty"
        ]
        try:
            for subproperty_extension in subproperty_extensions:
                code_extension = next(
                    e for e in subproperty_extension["extension"] if e["url"] == "code"
                )
                subcode = f"{code}.{code_extension['valueCode']}"
                if subcode not in IGNORED_PROPERTIES:
                    value_extension = next(
                        e
                        for e in subproperty_extension["extension"]
                        if e["url"] == "value"
                    )
                    value_key = next(
                        k for k in value_extension.keys() if k.startswith("value")
                    )
                    result[subcode] = value_extension[value_key]
                    add_subproperties_to_dict(subproperty_extension, subcode, result)
        except StopIteration:
            return
