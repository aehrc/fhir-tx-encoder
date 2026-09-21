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

# The recipes step: construction, prep, bake, tidy, required_pkgs and print.
#
# recipes is a suggested dependency, so the whole file is skipped when it is
# absent. The terminology server is answered from fixtures throughout, so no
# network access is required.

skip_if_not_installed("recipes")

test_scope <- "http://snomed.info/sct?fhir_vs=ecl/(%3E%3E%20363346000)"
test_tx_url <- "https://tx.example.org/fhir"

# Expansion order, which is also the order of the encoder's first six features.
expected_codes <- c(
  "404684003", "64572001", "363346000", "399981008", "55342001", "138875005"
)

# The property features the single-page fixture carries, in the byte order the
# encoder emits them.
expected_property_features <- c(
  "609096000.116676008=108369006",
  "609096000.116676008=1240414004",
  "609096000.116676008=400177003"
)

# Two in-scope codes: "Neoplasm and/or hamartoma" and "Malignant neoplastic
# disease". The scope holds six codes, so the training data deliberately covers
# only part of it.
training_data <- data.frame(
  code = c("399981008", "363346000"),
  outcome = c(0, 1),
  stringsAsFactors = FALSE
)

#' Build a recipe carrying the step, without prepping it.
new_test_recipe <- function(data = training_data, ...) {
  step_fhir_tx(
    recipes::recipe(outcome ~ code, data = data),
    code,
    scope = test_scope,
    tx_url = test_tx_url,
    ...
  )
}

#' Prep a recipe carrying the step, with the server answered from fixtures.
prep_test_recipe <- function(data = training_data, ...) {
  suppressMessages(with_fixture_api(
    recipes::prep(new_test_recipe(data, ...)),
    expand = "expand-single-page",
    closure_update = "closure-update"
  ))
}

#' Build the equivalent encoder through the functional API, for comparison.
build_reference_encoder <- function(...) {
  suppressMessages(with_fixture_api(
    fhir_tx_encoder(scope = test_scope, tx_url = test_tx_url, ...),
    expand = "expand-single-page",
    closure_update = "closure-update"
  ))
}

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

test_that("the step is added to the recipe untrained, carrying its parameters", {
  # Arrange / Act - no prep, so no terminology server contact.
  rec <- new_test_recipe(properties = "116676008", batch_size = 10)
  step <- rec$steps[[1]]

  # Assert - the step is recognisable and holds what it was given.
  expect_s3_class(step, "step_fhir_tx")
  expect_s3_class(step, "step")
  expect_false(step$trained)
  expect_null(step$encoder)
  expect_equal(step$scope, test_scope)
  expect_equal(step$tx_url, test_tx_url)
  expect_true(step$subsumption)
  expect_equal(step$properties, "116676008")
  expect_equal(step$batch_size, 10)
  expect_equal(step$role, "predictor")
  expect_false(step$skip)
  # The default id is generated from the step name.
  expect_match(step$id, "^fhir_tx_")
})

test_that("selecting no variable at all is rejected when the step is created", {
  # Arrange
  rec <- recipes::recipe(outcome ~ code, data = training_data)

  # Act / Assert - recipes' own selector check fires before anything else.
  expect_error(
    step_fhir_tx(rec, scope = test_scope, tx_url = test_tx_url),
    "at least one variable specification",
    fixed = TRUE
  )
})

test_that("step_fhir_tx() explains itself when recipes is not installed", {
  # Arrange - simulate recipes' absence by stubbing the package's own
  # availability check, which wraps requireNamespace(). Nothing is uninstalled.
  rec <- recipes::recipe(outcome ~ code, data = training_data)
  local_mocked_bindings(recipes_installed = function() FALSE)

  # Act / Assert
  expect_error(
    step_fhir_tx(rec, code, scope = test_scope),
    "The recipes package must be installed to use step_fhir_tx().",
    fixed = TRUE
  )
})

# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------

test_that("prep builds the encoder from the scope, not from the training data", {
  # Arrange / Act - the training data holds 2 codes; the scope holds 6.
  prepped <- prep_test_recipe()
  step <- prepped$steps[[1]]

  # Assert - the encoder covers the whole scope.
  expect_s3_class(step$encoder, "fhir_tx_encoder")
  expect_equal(step$encoder$codes, expected_codes)
  expect_equal(step$encoder$feature_names, expected_codes)
  # Assert - the step is now trained and knows its column.
  expect_true(step$trained)
  expect_equal(step$columns, "code")
  # Assert - the scope, not the training data, was sent to the server.
  expect_equal(request_query(expand_requests()[[1]], "url"), test_scope)
})

test_that("prep passes the step's options through to the encoder", {
  # Arrange / Act - subsumption off and one property requested.
  prepped <- prep_test_recipe(subsumption = FALSE, properties = "116676008")
  encoder <- prepped$steps[[1]]$encoder

  # Assert - the property widening happened and the closure was not used.
  expect_equal(
    encoder$feature_names, c(expected_codes, expected_property_features)
  )
  expect_length(closure_requests(), 0L)
  # Assert - the property was requested of the server.
  expect_equal(request_query(expand_requests()[[1]], "property"), "116676008")
})

test_that("selecting more than one column is a clear error", {
  # Arrange - two candidate code columns, both selected.
  data <- data.frame(
    code = "399981008", other = "363346000", outcome = 1,
    stringsAsFactors = FALSE
  )
  rec <- step_fhir_tx(
    recipes::recipe(outcome ~ code + other, data = data),
    recipes::all_predictors(),
    scope = test_scope,
    tx_url = test_tx_url
  )

  # Act / Assert - the error names the offending columns, and no request is
  # made (the fixture router is not even installed).
  expect_error(
    recipes::prep(rec),
    "step_fhir_tx() requires exactly one code column, but 2 were selected: code, other.",
    fixed = TRUE
  )
})

test_that("selecting nothing that exists is a clear error", {
  # Arrange - the only predictor is the character code column, so a numeric
  # selector matches nothing.
  rec <- step_fhir_tx(
    recipes::recipe(outcome ~ code, data = training_data),
    recipes::all_numeric_predictors(),
    scope = test_scope,
    tx_url = test_tx_url
  )

  # Act / Assert
  expect_error(
    recipes::prep(rec),
    "step_fhir_tx() requires exactly one code column, but none were selected.",
    fixed = TRUE
  )
})

# ---------------------------------------------------------------------------
# bake
# ---------------------------------------------------------------------------

test_that("bake replaces the code column with one numeric column per feature", {
  # Arrange
  prepped <- prep_test_recipe()

  # Act
  baked <- recipes::bake(prepped, new_data = training_data)

  # Assert - the code column is gone and the feature columns are in place,
  # named verbatim after the encoder's feature names.
  expect_s3_class(baked, "tbl_df")
  expect_false("code" %in% names(baked))
  expect_equal(names(baked), c("outcome", expected_codes))
  expect_true(all(vapply(baked[expected_codes], is.numeric, logical(1))))
  # Assert - the untouched columns survive.
  expect_equal(baked$outcome, training_data$outcome)
})

test_that("baked values equal the functional API's transform output", {
  # Arrange - the same fixtures drive both surfaces.
  prepped <- prep_test_recipe()
  encoder <- build_reference_encoder()

  # Act
  baked <- recipes::bake(prepped, new_data = training_data)
  expected <- as.matrix(
    transform(encoder, matrix(training_data$code, ncol = 1))
  )

  # Assert - cell for cell.
  expect_equal(unname(as.matrix(baked[expected_codes])), expected)
})

test_that("property features become their own baked columns", {
  # Arrange - the documented 6 x 9 widening.
  prepped <- prep_test_recipe(properties = "116676008")
  encoder <- build_reference_encoder(properties = "116676008")

  # Act
  baked <- recipes::bake(prepped, new_data = training_data)
  expected <- as.matrix(
    transform(encoder, matrix(training_data$code, ncol = 1))
  )

  # Assert - nine feature columns, values matching the functional API.
  expect_equal(
    names(baked),
    c("outcome", expected_codes, expected_property_features)
  )
  expect_equal(
    unname(as.matrix(baked[c(expected_codes, expected_property_features)])),
    expected
  )
})

test_that("a factor code column is coerced to character", {
  # Arrange - the same codes, but as a factor.
  data <- data.frame(
    code = factor(c("399981008", "363346000")),
    outcome = c(0, 1)
  )
  prepped <- prep_test_recipe(data = data)
  encoder <- build_reference_encoder()

  # Act
  baked <- recipes::bake(prepped, new_data = data)

  # Assert - the factor labels, not the integer codes, were encoded.
  expect_equal(
    unname(as.matrix(baked[expected_codes])),
    as.matrix(transform(encoder, matrix(as.character(data$code), ncol = 1)))
  )
})

test_that("baking new data works and preserves row order", {
  # Arrange - new data in a different order, with a code the training data
  # never contained.
  prepped <- prep_test_recipe()
  encoder <- build_reference_encoder()
  new_data <- data.frame(
    code = c("138875005", "404684003", "363346000"),
    stringsAsFactors = FALSE
  )

  # Act
  baked <- recipes::bake(prepped, new_data = new_data)

  # Assert
  expect_equal(nrow(baked), 3L)
  expect_equal(
    unname(as.matrix(baked[expected_codes])),
    as.matrix(transform(encoder, matrix(new_data$code, ncol = 1)))
  )
})

test_that("an out-of-scope code at bake time raises the encoder's error", {
  # Arrange
  prepped <- prep_test_recipe()
  new_data <- data.frame(code = "999999999", outcome = 0)

  # Act / Assert - the same wording the functional API uses.
  expect_error(
    recipes::bake(prepped, new_data = new_data),
    "Encountered code not in scope: 999999999",
    fixed = TRUE
  )
})

# ---------------------------------------------------------------------------
# tidy, required_pkgs and print
# ---------------------------------------------------------------------------

test_that("tidy reports the selector before prep and the column after", {
  # Arrange
  rec <- new_test_recipe(properties = "116676008", batch_size = 10)

  # Act
  untrained <- recipes::tidy(rec, number = 1)

  # Assert - a one-row tibble of the step's parameters.
  expect_s3_class(untrained, "tbl_df")
  expect_equal(nrow(untrained), 1L)
  expect_equal(untrained$terms, "code")
  expect_equal(untrained$scope, test_scope)
  expect_equal(untrained$tx_url, test_tx_url)
  expect_true(untrained$subsumption)
  expect_equal(untrained$properties, list("116676008"))
  expect_equal(untrained$batch_size, 10)
  expect_equal(untrained$id, rec$steps[[1]]$id)

  # Act - after prep the resolved column name is reported instead.
  trained <- recipes::tidy(prep_test_recipe(), number = 1)

  # Assert
  expect_equal(trained$terms, "code")
  expect_equal(trained$properties, list(NULL))
})

test_that("required_pkgs names this package", {
  # Arrange / Act / Assert
  expect_equal(
    recipes::required_pkgs(new_test_recipe()$steps[[1]]), "fhirtxencoder"
  )
})

test_that("print gives a recipes-style one-line summary", {
  # Arrange - recipes prints steps through cli, which writes to stderr.
  rec <- new_test_recipe()

  # Act
  untrained <- paste(capture_messages(print(rec$steps[[1]])), collapse = "")
  trained <- paste(
    capture_messages(print(prep_test_recipe()$steps[[1]])),
    collapse = ""
  )

  # Assert - one line naming the operation and the column, marked Trained once
  # the step has been prepped.
  expect_match(untrained, "FHIR terminology encoding for: code", fixed = TRUE)
  expect_false(grepl("Trained", untrained, fixed = TRUE))
  expect_match(trained, "FHIR terminology encoding for: code", fixed = TRUE)
  expect_match(trained, "Trained", fixed = TRUE)
  # Assert - the step is returned invisibly, as print methods must.
  expect_invisible(suppressMessages(print(rec$steps[[1]])))
})
