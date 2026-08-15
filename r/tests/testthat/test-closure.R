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

# The `$closure` operation wrappers: what is posted and what is read back out
# of the returned ConceptMap.

test_tx_url <- "https://tx.example.org/fhir"
test_name <- "0f3b1c2d4e5f6a7b8c9d0e1f2a3b4c5d"

# Two codings that between them cover both the "no version" and "version"
# cases, and both of the fields that must never be sent (display, extension).
test_codings <- list(
  list(
    system = "http://snomed.info/sct",
    code = "363346000",
    display = "Malignant neoplastic disease",
    extension = list(list(url = "http://example.org/ext", valueCode = "x"))
  ),
  list(
    system = "http://snomed.info/sct",
    version = "http://snomed.info/sct/32506021000036107/version/20260731",
    code = "64572001",
    display = "Disease"
  )
)

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

test_that("initialisation posts a Parameters body carrying a UUID hex name", {
  # Arrange / Act
  name <- with_fixture_api(closure_initialize(test_tx_url))

  # Assert - the returned name is a bare UUID with the hyphens removed, which
  # is what Python's `uuid4().hex` produces.
  expect_match(name, "^[0-9a-f]{32}$")

  # Assert - a single POST to $closure with just the name parameter.
  requests <- closure_requests("initialize")
  expect_length(requests, 1L)
  expect_equal(requests[[1]]$method, "POST")
  expect_match(requests[[1]]$path, "\\$closure$")
  expect_equal(requests[[1]]$body$resourceType, "Parameters")
  expect_length(requests[[1]]$body$parameter, 1L)
  expect_equal(closure_name(requests[[1]]), name)
  expect_equal(requests[[1]]$headers$Accept, "application/fhir+json")
})

test_that("each initialisation uses a fresh closure table name", {
  # Arrange / Act - two initialisations within one mocked session.
  names <- with_fixture_api(
    c(closure_initialize(test_tx_url), closure_initialize(test_tx_url)),
    closure_initialize = c("closure-initialize", "closure-initialize")
  )

  # Assert - names must differ so that concurrent encoders do not share a
  # server-side closure table.
  expect_length(unique(names), 2L)
})

test_that("the table name is independent of the caller's seed and leaves it untouched", {
  # Arrange - a caller who seeded the generator for a reproducible workflow.
  set.seed(42)
  before <- get(".Random.seed", envir = globalenv())

  # Act - two constructions from the identical seeded state.
  first <- with_fixture_api(closure_initialize(test_tx_url))
  after <- get(".Random.seed", envir = globalenv())
  set.seed(42)
  second <- with_fixture_api(closure_initialize(test_tx_url))

  # Assert - the names differ, so two callers who seeded alike do not collide
  # on a shared server-side closure table.
  expect_false(identical(first, second))
  # Assert - the caller's random stream is exactly where they left it, so the
  # rest of their seeded workflow is unaffected.
  expect_identical(after, before)
})

test_that("a non-2xx initialisation response raises an error carrying the status", {
  # Arrange / Act / Assert
  expect_error(
    with_fixture_api(
      closure_initialize(test_tx_url),
      closure_initialize = fixture_response("closure-initialize", status = 500L)
    ),
    class = "httr2_http_500"
  )
})

# ---------------------------------------------------------------------------
# Update request body
# ---------------------------------------------------------------------------

test_that("an update posts a flat parameter array of name plus one concept per coding", {
  # Arrange
  codings <- read_fixture("expand-single-page")$expansion$contains

  # Act
  with_fixture_api(
    closure_update(test_tx_url, test_name, codings),
    closure_update = "closure-update"
  )

  # Assert - the array is flat: no nested sub-array, so every entry is a
  # parameter with a name of its own. This is the deliberate correction of the
  # Python implementation's invalid nesting.
  record <- closure_requests("update")[[1]]
  parameters <- record$body$parameter
  expect_equal(record$body$resourceType, "Parameters")
  expect_length(parameters, 1L + length(codings))
  expect_true(all(vapply(
    parameters, function(p) is.character(p$name), logical(1)
  )))

  # Assert - the name comes first, then the concepts in coding order.
  expect_equal(parameters[[1]]$name, "name")
  expect_equal(closure_name(record), test_name)
  concepts <- closure_concepts(record)
  expect_length(concepts, 6L)
  expect_equal(concepts[[1]]$code, "404684003")
  expect_equal(concepts[[6]]$code, "138875005")
})

test_that("update codings carry only system, version and code, and only when present", {
  # Arrange / Act
  with_fixture_api(
    closure_update(test_tx_url, test_name, test_codings),
    closure_update = "closure-update-no-group"
  )

  # Assert - display and extension are dropped; version appears only for the
  # coding that had one.
  concepts <- closure_concepts(closure_requests("update")[[1]])
  expect_equal(names(concepts[[1]]), c("system", "code"))
  expect_equal(names(concepts[[2]]), c("system", "version", "code"))
  expect_equal(
    concepts[[2]]$version,
    "http://snomed.info/sct/32506021000036107/version/20260731"
  )
})

# ---------------------------------------------------------------------------
# Pair extraction
# ---------------------------------------------------------------------------

test_that("only targets with an equivalence of subsumes become pairs", {
  # Arrange / Act - the fixture holds 15 `subsumes` targets plus one `equal`
  # and one `specializes` target that must be ignored.
  pairs <- with_fixture_api(
    closure_update(test_tx_url, test_name, test_codings),
    closure_update = "closure-update"
  )

  # Assert - a pair is (element.code, target.code), in document order.
  expect_length(pairs$source, 15L)
  expect_length(pairs$target, 15L)
  expect_equal(pairs$source[[1]], "399981008")
  expect_equal(pairs$target[[1]], "138875005")

  # Assert - the `equal` target of 404684003 is excluded.
  expect_false(any(pairs$source == "404684003" & pairs$target == "55342001"))
  # Assert - 138875005's only target is `specializes`, so it contributes none.
  expect_false(any(pairs$source == "138875005"))
})

test_that("a response with no group yields zero pairs", {
  # Arrange / Act - a ConceptMap with no `group` element at all.
  pairs <- with_fixture_api(
    closure_update(test_tx_url, test_name, test_codings),
    closure_update = "closure-update-no-group"
  )

  # Assert - empty character vectors, not NULL, so callers need no special case.
  expect_equal(pairs$source, character())
  expect_equal(pairs$target, character())
})
