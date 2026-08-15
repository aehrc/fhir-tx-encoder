# How to contribute

Thanks for your interest in contributing to "fhir-tx-encoder".

You can find out a bit more about the project by reading the [README](README.md)
file within this repository.

## Repository layout

The repository houses two implementations of the same encoder:

| Directory | Package | Language |
| --------- | ------- | -------- |
| [`python/`](python) | `fhir-tx-encoder` (PyPI) | Python |
| [`r/`](r) | `fhirtxencoder` (GitHub) | R |

The community documents (this file, `README.md`, `LICENSE`,
`CODE_OF_CONDUCT.md`) live at the root and cover both.

### Why `python/LICENSE` is a symlink

`python/LICENSE` is a symbolic link to the repository's root `LICENSE`. It is
not a stray duplicate, and it should not be replaced with a copy or deleted.

`python/setup.py` uses setuptools' default licence-file discovery, which only
looks inside the package directory. Before the Python package moved from the
repository root into `python/`, `LICENSE` sat next to `setup.py` and setuptools
packaged it as `fhir_tx_encoder-<version>.dist-info/licenses/LICENSE`. The
symlink keeps that true after the move, so the published wheel and sdist remain
byte-identical to earlier releases. Removing it would silently drop the licence
from the distributed artefact.

Verify with:

```bash
cd python
uv venv --python 3.11 && uv pip install build && uv run python -m build
unzip -p dist/*.whl 'fhir_tx_encoder-*.dist-info/RECORD' | grep LICENSE
```

## Running the checks

Python:

```bash
cd python
uv venv --python 3.11
uv run python -m build
```

Python 3.11 is pinned here because `setup.py` requires `scipy~=1.11.3`, which
publishes no wheels for later Python versions.

R (see [`r/README.md`](r/README.md) for detail):

```bash
Rscript -e 'devtools::test("r")'
R CMD build r && R CMD check --as-cran fhirtxencoder_1.0.0.tar.gz
Rscript -e 'print(covr::package_coverage("r"))'
```

The R unit suite answers every terminology server request from canned fixtures,
so it needs no network. `r/tests/testthat/test-integration.R` is the exception:
it runs the README example against the live CSIRO public Ontoserver, and is
skipped when `NOT_CRAN` is unset, when the machine is offline, or when `CI` is
set.

## Reporting issues

Issues can be used to:

* Report a defect
* Request a new feature or enhancement
* Ask a question

## Creating a pull request

Please communicate with us (preferably through creation of an issue) before
embarking on any significant work within a pull request. This will prevent
situations where people are working at cross-purposes.

Your branch should be named `issue/[GitHub issue #]`.

### Coding conventions

Python code in `python/` uses [Black](https://github.com/psf/black), please use
it to reformat your code before pushing.

R code in `r/` follows the [tidyverse style guide](https://style.tidyverse.org/)
with an 80 character line limit. Every exported and internal function carries a
roxygen2 block; regenerate `NAMESPACE` and `man/` with
`Rscript -e 'roxygen2::roxygenise("r")'` after changing documentation. New
behaviour needs a test - the R package is held at full line coverage.

## Code of conduct

Before making a contribution, please read the
[code of conduct](CODE_OF_CONDUCT.md).
