import uuid

import requests


class Closure:
    """
    Builds a table with two columns: source and target. Each row indicates that the source code
    is subsumed by the target code.

    The table is not kept in memory, this class simply returns the new entries. This is so that the
    entries can be processed without holding the entire table in memory.
    """

    def __init__(self, tx_url: str):
        self._tx_url = tx_url
        self._name = self._initialize()

    def _initialize(self):
        name = uuid.uuid4().hex
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
        execution_request = {
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
        execution_response = requests.post(
            f"{self._tx_url}/$closure",
            json=execution_request,
            headers={
                "Content-Type": "application/fhir+json",
                "Accept": "application/fhir+json",
            },
        )
        execution_response.raise_for_status()
        concept_map = execution_response.json()
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
