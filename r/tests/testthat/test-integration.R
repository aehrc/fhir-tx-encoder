#
#     Copyright © 2026, Commonwealth Scientific and Industrial Research
#     Organisation (CSIRO) ABN 41 687 119 230.
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
#     Author: John Grimes
#

# Live equivalence check against a real terminology server.
#
# Every other test in this suite answers from canned fixtures. This one is the
# cross-language acceptance evidence: it runs the example documented in the
# README against the CSIRO public Ontoserver and asserts the same 2x9 matrix
# and the same nine feature names the Python library produces.
#
# It is skipped when:
#
# - running on CRAN, which forbids network access during checks;
# - the machine has no network;
# - the `CI` environment variable is set. Continuous integration sets `CI` (as
#   GitHub Actions does), so the scheduled check never fails because of a
#   terminology server outage or a content change on the server. Running the
#   suite locally leaves `CI` unset, so the test runs by default.

TX_HOST <- "tx.ontoserver.csiro.au"
TX_URL <- paste0("https://", TX_HOST, "/fhir")

# Ancestors of the SNOMED CT concept "Malignant neoplastic disease" (363346000).
LIVE_SCOPE <- "http://snomed.info/sct?fhir_vs=ecl/(%3E%3E%20363346000)"

#' Skip unless a live terminology server can be reached and used.
#'
#' @return Invisibly NULL, or aborts the calling test with a skip.
skip_unless_live <- function() {
  testthat::skip_on_cran()
  testthat::skip_if_offline(host = TX_HOST)
  testthat::skip_if(
    nzchar(Sys.getenv("CI")),
    "live terminology server tests do not run in CI"
  )
}

# SC-001: the README example, run against the live CSIRO public server, must
# yield the 2x9 matrix and feature names documented for the Python library.
test_that("the README example reproduces the documented Python output", {
  skip_unless_live()

  # Arrange: the documented scope, with "Associated morphology" (116676008)
  # requested as a property.
  encoder <- suppressMessages(fhir_tx_encoder(
    scope = LIVE_SCOPE,
    tx_url = TX_URL,
    properties = "116676008"
  ))

  # Act: encode the same two codes the README encodes.
  result <- transform(
    encoder, matrix(c("399981008", "363346000"), ncol = 1)
  )

  # Assert: the exact shape, cells and feature names.
  expect_equal(dim(result), c(2L, 9L))
  expect_equal(
    unname(as.matrix(result)),
    matrix(
      c(
        1, 1, 0, 1, 0, 1, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 0, 1, 0
      ),
      nrow = 2, ncol = 9, byrow = TRUE
    )
  )
  expect_equal(
    encoder$feature_names,
    c(
      "404684003", "64572001", "363346000", "399981008", "55342001",
      "138875005", "609096000.116676008=108369006",
      "609096000.116676008=1240414004", "609096000.116676008=400177003"
    )
  )

  # The first six features are the codes of the scope, and their displays are
  # reported in the same order.
  expect_equal(encoder$codes, encoder$feature_names[1:6])
  expect_equal(
    encoder$displays,
    c(
      "Clinical finding", "Disease", "Malignant neoplastic disease",
      "Neoplasm and/or hamartoma", "Neoplastic disease", "SNOMED CT Concept"
    )
  )
})

# Construction reports its progress, as the Python library does.
test_that("construction against the live server reports progress", {
  skip_unless_live()

  expect_message(
    fhir_tx_encoder(scope = LIVE_SCOPE, tx_url = TX_URL),
    "Expanding value set"
  )
})

# An out-of-scope code is rejected against a live scope, not only a mocked one.
test_that("a code outside the live scope is rejected", {
  skip_unless_live()

  encoder <- suppressMessages(
    fhir_tx_encoder(scope = LIVE_SCOPE, tx_url = TX_URL)
  )

  expect_error(
    transform(encoder, matrix("74400008", ncol = 1)),
    "Encountered code not in scope: 74400008",
    fixed = TRUE
  )
})
