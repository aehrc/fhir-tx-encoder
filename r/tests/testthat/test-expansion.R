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

# Paged `ValueSet/$expand` behaviour: what is requested, what is accumulated
# and what is reported.

test_scope <- "http://snomed.info/sct?fhir_vs=ecl/(%3E%3E%20363346000)"
test_tx_url <- "https://tx.example.org/fhir"

# The six concepts of the captured scope, in expansion order.
expected_codes <- c(
  "404684003", "64572001", "363346000", "399981008", "55342001", "138875005"
)
expected_displays <- c(
  "Clinical finding", "Disease", "Malignant neoplastic disease",
  "Neoplasm and/or hamartoma", "Neoplastic disease", "SNOMED CT Concept"
)

# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

test_that("a single-page expansion yields its codes, displays and codings", {
  # Arrange / Act - one page holds the whole scope.
  expansion <- suppressMessages(with_fixture_api(
    expand_scope(test_scope, test_tx_url, properties = NULL, batch_size = 50000),
    expand = "expand-single-page"
  ))

  # Assert - codes and displays are in expansion order, and the raw codings
  # are retained in a single batch for the later closure update.
  expect_equal(expansion$codes, expected_codes)
  expect_equal(expansion$displays, expected_displays)
  expect_length(expansion$coding_batches, 1L)
  expect_length(expansion$coding_batches[[1]], 6L)
  expect_equal(expansion$coding_batches[[1]][[1]]$code, "404684003")
  expect_equal(
    expansion$coding_batches[[1]][[1]]$system, "http://snomed.info/sct"
  )
})

test_that("a multi-page expansion pages by offset and accumulates in order", {
  # Arrange / Act - a batch size of 4 over a total of 6 needs two requests.
  expansion <- suppressMessages(with_fixture_api(
    expand_scope(test_scope, test_tx_url, properties = NULL, batch_size = 4),
    expand = c("0" = "expand-page-1", "4" = "expand-page-2")
  ))

  # Assert - offsets advance by the batch size, and the loop stops once the
  # offset passes the reported total (0, then 4, then 8 > 6).
  requests <- expand_requests()
  expect_length(requests, 2L)
  expect_equal(request_query(requests[[1]], "offset"), "0")
  expect_equal(request_query(requests[[2]], "offset"), "4")

  # Assert - the pages concatenate in request order.
  expect_equal(expansion$codes, expected_codes)
  expect_equal(expansion$displays, expected_displays)
  expect_length(expansion$coding_batches, 2L)
  expect_length(expansion$coding_batches[[1]], 4L)
  expect_length(expansion$coding_batches[[2]], 2L)
})

test_that("a single-page expansion yields a property list per concept", {
  # Arrange / Act - three of the six concepts in this fixture carry an
  # associated morphology property nested in a role group.
  expansion <- suppressMessages(with_fixture_api(
    expand_scope(
      test_scope, test_tx_url,
      properties = "116676008", batch_size = 50000
    ),
    expand = "expand-single-page"
  ))

  # Assert - one list per concept, aligned with the codes, so that the
  # vectorised properties can be bound straight onto the encoding.
  expect_equal(
    expansion$properties,
    list(
      list(),
      list(),
      list("609096000.116676008" = "1240414004"),
      list("609096000.116676008" = "400177003"),
      list("609096000.116676008" = "108369006"),
      list()
    )
  )
})

test_that("property lists accumulate across pages in expansion order", {
  # Arrange / Act - the same six concepts, split over two pages.
  expansion <- suppressMessages(with_fixture_api(
    expand_scope(
      test_scope, test_tx_url,
      properties = "116676008", batch_size = 4
    ),
    expand = c("0" = "expand-page-1", "4" = "expand-page-2")
  ))

  # Assert - paging does not disturb the alignment between codes and
  # properties.
  expect_equal(expansion$codes, expected_codes)
  expect_equal(
    expansion$properties,
    list(
      list(),
      list(),
      list("609096000.116676008" = "1240414004"),
      list("609096000.116676008" = "400177003"),
      list("609096000.116676008" = "108369006"),
      list()
    )
  )
})

test_that("a concept with no display is recorded as NA", {
  # Arrange / Act - the fifth concept of this fixture has no `display`.
  expansion <- suppressMessages(with_fixture_api(
    expand_scope(test_scope, test_tx_url, properties = NULL, batch_size = 50000),
    expand = "expand-missing-display"
  ))

  # Assert - the gap is NA rather than a dropped or empty entry, so displays
  # stay aligned with codes.
  expect_length(expansion$displays, 6L)
  expect_identical(expansion$displays[[5]], NA_character_)
  expect_identical(expansion$displays[[1]], "Clinical finding")
  expect_equal(expansion$codes, expected_codes)
})

# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------

test_that("the expansion request carries url, count, offset and an Accept header", {
  # Arrange / Act
  suppressMessages(with_fixture_api(
    expand_scope(test_scope, test_tx_url, properties = NULL, batch_size = 50000),
    expand = "expand-single-page"
  ))

  # Assert
  request <- expand_requests()[[1]]
  expect_equal(request$method, "GET")
  expect_match(request$path, "ValueSet/\\$expand$")
  expect_equal(request_query(request, "url"), test_scope)
  expect_equal(request_query(request, "count"), "50000")
  expect_equal(request_query(request, "offset"), "0")
  expect_equal(request$headers$Accept, "application/fhir+json")
})

test_that("no property parameter is sent when no properties are requested", {
  # Arrange / Act
  suppressMessages(with_fixture_api(
    expand_scope(test_scope, test_tx_url, properties = NULL, batch_size = 50000),
    expand = "expand-single-page"
  ))

  # Assert
  expect_equal(request_query(expand_requests()[[1]], "property"), character())
})

test_that("one property parameter is sent per requested property code", {
  # Arrange / Act - two property codes are requested.
  suppressMessages(with_fixture_api(
    expand_scope(
      test_scope, test_tx_url,
      properties = c("116676008", "609096000"), batch_size = 50000
    ),
    expand = "expand-single-page"
  ))

  # Assert - the parameter is repeated rather than comma-joined.
  expect_equal(
    request_query(expand_requests()[[1]], "property"),
    c("116676008", "609096000")
  )
})

# ---------------------------------------------------------------------------
# Errors and progress
# ---------------------------------------------------------------------------

test_that("an empty expansion is an error naming the scope", {
  # Arrange / Act / Assert - the fixture reports total 0 and no `contains`.
  expect_error(
    suppressMessages(with_fixture_api(
      expand_scope(test_scope, test_tx_url, properties = NULL, batch_size = 50000),
      expand = "expand-empty"
    )),
    paste0("Value set expansion is empty: ", test_scope),
    fixed = TRUE
  )
})

test_that("a non-2xx expansion response raises an error carrying the status", {
  # Arrange / Act / Assert - httr2 turns the 404 into a typed condition.
  expect_error(
    suppressMessages(with_fixture_api(
      expand_scope(test_scope, test_tx_url, properties = NULL, batch_size = 50000),
      expand = fixture_response("expand-empty", status = 404L)
    )),
    class = "httr2_http_404"
  )
})

test_that("expansion progress is reported through suppressible messages", {
  # Arrange / Act - capture the messages emitted over a two-page expansion.
  emitted <- paste(capture_messages(with_fixture_api(
    expand_scope(test_scope, test_tx_url, properties = NULL, batch_size = 4),
    expand = c("0" = "expand-page-1", "4" = "expand-page-2")
  )), collapse = "")

  # Assert - the scope, one line per page and a completion line, mirroring the
  # Python implementation's output.
  expect_match(
    emitted, paste0("Expanding value set: ", test_scope),
    fixed = TRUE
  )
  expect_match(emitted, "Expanding (4 items, offset 0, total 6)", fixed = TRUE)
  expect_match(emitted, "Expanding (2 items, offset 4, total 6)", fixed = TRUE)
  expect_match(emitted, "Expansion complete", fixed = TRUE)
})
