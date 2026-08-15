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

# Construction of the encoder and the transform and print methods.

test_scope <- "http://snomed.info/sct?fhir_vs=ecl/(%3E%3E%20363346000)"
test_tx_url <- "https://tx.example.org/fhir"

expected_codes <- c(
  "404684003", "64572001", "363346000", "399981008", "55342001", "138875005"
)

# The documented encoding for the captured scope: the identity, plus a 1 at
# [row(source), col(target)] for every subsumption pair. Row order and column
# order are both expansion order.
expected_encoding <- function() {
  matrix(
    c(
      #  404684003 64572001 363346000 399981008 55342001 138875005
      1, 0, 0, 0, 0, 1, # 404684003
      1, 1, 0, 0, 0, 1, # 64572001
      1, 1, 1, 1, 1, 1, # 363346000
      1, 1, 0, 1, 0, 1, # 399981008
      1, 1, 0, 1, 1, 1, # 55342001
      0, 0, 0, 0, 0, 1 # 138875005
    ),
    nrow = 6, byrow = TRUE
  )
}

# Build the encoder used by most tests: the whole scope in one expansion page
# and one closure update.
build_test_encoder <- function(subsumption = TRUE) {
  suppressMessages(with_fixture_api(
    fhir_tx_encoder(
      scope = test_scope, tx_url = test_tx_url, subsumption = subsumption
    ),
    expand = "expand-single-page",
    closure_update = "closure-update"
  ))
}

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

test_that("construction produces the documented identity plus subsumption matrix", {
  # Arrange / Act
  encoder <- build_test_encoder()

  # Assert - sparse, 6 x 6, and exactly the documented cells.
  expect_s4_class(encoder$encoded, "dgCMatrix")
  expect_equal(dim(encoder$encoded), c(6L, 6L))
  expect_equal(as.matrix(encoder$encoded), expected_encoding())
})

test_that("construction records codes, displays, feature names, index and scope", {
  # Arrange / Act
  encoder <- build_test_encoder()

  # Assert
  expect_s3_class(encoder, "fhir_tx_encoder")
  expect_equal(encoder$codes, expected_codes)
  expect_equal(
    encoder$displays,
    c(
      "Clinical finding", "Disease", "Malignant neoplastic disease",
      "Neoplasm and/or hamartoma", "Neoplastic disease", "SNOMED CT Concept"
    )
  )
  # Without properties, the features are just the codes.
  expect_equal(encoder$feature_names, expected_codes)
  # The index is 1-based, matching R's row and column numbering.
  expect_equal(
    encoder$index,
    stats::setNames(seq_along(expected_codes), expected_codes)
  )
  expect_equal(encoder$scope, test_scope)
})

test_that("subsumption = FALSE yields the identity and makes no closure requests", {
  # Arrange / Act
  encoder <- build_test_encoder(subsumption = FALSE)

  # Assert - a pure one-hot encoding.
  expect_equal(as.matrix(encoder$encoded), diag(6))
  # Assert - the closure operation is not called at all.
  expect_length(closure_requests(), 0L)
})

test_that("a paged scope accumulates the same matrix over one closure update per batch", {
  # Arrange / Act - two expansion pages of 4 and 2 concepts.
  encoder <- suppressMessages(with_fixture_api(
    fhir_tx_encoder(scope = test_scope, tx_url = test_tx_url, batch_size = 4),
    expand = c("0" = "expand-page-1", "4" = "expand-page-2"),
    closure_update = c("closure-update-batch-1", "closure-update-batch-2")
  ))

  # Assert - the batched result matches the single-batch result.
  expect_equal(as.matrix(encoder$encoded), expected_encoding())

  # Assert - one update per expansion batch, carrying that batch's codings.
  updates <- closure_requests("update")
  expect_length(updates, 2L)
  expect_length(closure_concepts(updates[[1]]), 4L)
  expect_length(closure_concepts(updates[[2]]), 2L)

  # Assert - every request names the same server-side closure table.
  initialised <- closure_name(closure_requests("initialize")[[1]])
  expect_equal(closure_name(updates[[1]]), initialised)
  expect_equal(closure_name(updates[[2]]), initialised)
})

test_that("an empty expansion fails construction with the scope named", {
  # Arrange / Act / Assert
  expect_error(
    suppressMessages(with_fixture_api(
      fhir_tx_encoder(scope = test_scope, tx_url = test_tx_url),
      expand = "expand-empty"
    )),
    paste0("Value set expansion is empty: ", test_scope),
    fixed = TRUE
  )
})

test_that("construction progress is reported through suppressible messages", {
  # Arrange / Act
  emitted <- paste(capture_messages(with_fixture_api(
    fhir_tx_encoder(scope = test_scope, tx_url = test_tx_url),
    expand = "expand-single-page",
    closure_update = "closure-update"
  )), collapse = "")

  # Assert - the same progress landmarks the Python implementation prints.
  expect_match(emitted, "Generating one-hot encoding... (6, 6)", fixed = TRUE)
  expect_match(emitted, "Creating index... 6 items", fixed = TRUE)
  expect_match(emitted, "Applying transitive closure...", fixed = TRUE)
  expect_match(emitted, "Batch 1 of 1, 6 items... 15 pairs added", fixed = TRUE)
  expect_match(
    emitted, "Subsumption encoding complete: (6, 6)",
    fixed = TRUE
  )
})

# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------

test_that("a single column of codes transforms to the rows of the encoding", {
  # Arrange - the README example: "Neoplasm and/or hamartoma" then "Malignant
  # neoplastic disease".
  encoder <- build_test_encoder()
  x <- matrix(c("399981008", "363346000"), ncol = 1)

  # Act
  result <- transform(encoder, x)

  # Assert
  expect_s4_class(result, "dgCMatrix")
  expect_equal(dim(result), c(2L, 6L))
  expect_equal(
    as.matrix(result),
    matrix(
      c(
        1, 1, 0, 1, 0, 1,
        1, 1, 1, 1, 1, 1
      ),
      nrow = 2, byrow = TRUE
    )
  )
})

test_that("multiple columns are encoded side by side in column order", {
  # Arrange - column 1 holds rows 4 and 3 of the encoding, column 2 holds rows
  # 6 and 1.
  encoder <- build_test_encoder()
  x <- matrix(
    c("399981008", "363346000", "138875005", "404684003"),
    nrow = 2
  )

  # Act
  result <- transform(encoder, X = x)

  # Assert - 2 x (2 columns * 6 features).
  expect_equal(dim(result), c(2L, 12L))
  full <- expected_encoding()
  expect_equal(
    as.matrix(result),
    cbind(full[c(4, 3), ], full[c(6, 1), ])
  )
})

test_that("a code outside the scope is an error naming that code", {
  # Arrange
  encoder <- build_test_encoder()

  # Act / Assert
  expect_error(
    transform(encoder, matrix(c("404684003", "999999999"), ncol = 1)),
    "Encountered code not in scope: 999999999",
    fixed = TRUE
  )
})

test_that("input that is not a two-dimensional character matrix is rejected", {
  # Arrange
  encoder <- build_test_encoder()
  expected <- "X must be a two-dimensional character matrix"

  # Act / Assert - a plain vector has no dimensions.
  expect_error(transform(encoder, c("404684003")), expected, fixed = TRUE)
  # A data frame is not a matrix.
  expect_error(
    transform(encoder, data.frame(code = "404684003")), expected,
    fixed = TRUE
  )
  # A three-dimensional array has the wrong number of dimensions.
  expect_error(
    transform(encoder, array("404684003", dim = c(2, 2, 2))), expected,
    fixed = TRUE
  )
  # A numeric matrix holds no codes.
  expect_error(
    transform(encoder, matrix(1:4, nrow = 2)), expected,
    fixed = TRUE
  )
})

# ---------------------------------------------------------------------------
# print
# ---------------------------------------------------------------------------

test_that("print summarises the scope and dimensions and returns the encoder invisibly", {
  # Arrange
  encoder <- build_test_encoder()

  # Act
  expect_output(printed <- withVisible(print(encoder)), test_scope, fixed = TRUE)

  # Assert - the summary names the counts, and the object comes back
  # unchanged and invisible.
  expect_output(print(encoder), "6 codes", fixed = TRUE)
  expect_output(print(encoder), "6 features", fixed = TRUE)
  expect_identical(printed$value, encoder)
  expect_false(printed$visible)
})
