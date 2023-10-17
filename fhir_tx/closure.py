import uuid

import requests


class Closure:
    """
    Uses a FHIR terminology server to incrementally build a transitive closure table that describes
    subsumption relationships between codings.

    The table is not retained, this class simply returns the new entries. This is so that the
    entries can be processed without ever having to hold the entire table in memory.
    """

    def __init__(self, tx_url: str):
        self._tx_url = tx_url
        self._name = self._initialize()

    def _initialize(self):
        # Create a unique name for the closure table, this is used in subsequent requests to tell
        # the server which closure we would like to update.
        name = uuid.uuid4().hex

        # The initialization request establishes the named closure table, ready for subsequent
        # update requests.
        initialize_request = {
            "resourceType": "Parameters",
            "parameter": [
                {
                    "name": "name",
                    "valueString": name,
                }
            ],
        }
        initialize_response = requests.post(
            f"{self._tx_url}/$closure",
            json=initialize_request,
            headers={
                "Content-Type": "application/fhir+json",
                "Accept": "application/fhir+json",
            },
        )
        initialize_response.raise_for_status()
        return name

    def update(self, codings: list[dict]) -> list[(str, str)]:
        # The update request adds a batch of codings to the closure table and returns the new
        # subsumption relationships.
        update_request = {
            "resourceType": "Parameters",
            "parameter": [
                {
                    "name": "name",
                    "valueString": self._name,
                },
                [
                    {
                        "name": "concept",
                        "valueCoding": dict(
                            (k, coding[k])
                            for k in ["system", "version", "code"]
                            if k in coding
                        ),
                    }
                    for coding in codings
                    if coding is not None
                ],
            ],
        }
        update_response = requests.post(
            f"{self._tx_url}/$closure",
            json=update_request,
            headers={
                "Content-Type": "application/fhir+json",
                "Accept": "application/fhir+json",
            },
        )
        update_response.raise_for_status()
        concept_map = update_response.json()

        # Convert the concept map into a list of tuples of source codes that subsume target codes.
        return (
            [
                (element["code"], target["code"])
                for group in concept_map["group"]
                for element in group["element"]
                for target in element["target"]
                if target["equivalence"] == "subsumes"
            ]
            if "group" in concept_map
            else []
        )
