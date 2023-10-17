from setuptools import setup

setup(
    name="fhir_tx_encoder",
    version="0.0.1",
    description="Tools for encoding FHIR terminology concepts for machine learning",
    author="Australian e-Health Research Centre, CSIRO",
    author_email="ontoserver-support@csiro.au",
    license="Apache 2.0",
    packages=["fhir_tx"],
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "requests",
        "scikit-learn",
    ],
)
