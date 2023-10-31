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

from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="fhir_tx_encoder",
    version="1.0.2",
    description="Tools for encoding FHIR terminology concepts for machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Australian e-Health Research Centre, CSIRO",
    author_email="ontoserver-support@csiro.au",
    license="Apache 2.0",
    packages=["fhir_tx"],
    install_requires=[
        "numpy~=1.26.0",
        "pandas~=2.1.2",
        "scipy~=1.11.3",
        "requests~=2.31.0",
        "scikit-learn~=1.3.2",
    ],
    python_requires=">=3.9",
)
