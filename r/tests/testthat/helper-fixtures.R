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

# Offline mock routing for the terminology server.
#
# The plan called for httptest2; it is not used, and is not a dependency of
# this package. Requests are instead intercepted with the same hook httptest2
# installs (`options(httr2_mock = ...)`, driven here through
# `httr2::with_mocked_responses()`) and answered from the canned FHIR JSON in
# `fixtures/`. httptest2's own `with_mock_api()` does not fit because it
# derives the mock file name from a hash of the request body, and the
# `$closure` request body carries a freshly generated UUID on every run, so no
# stable file name exists for it.
#
# Every intercepted request is recorded so that tests can assert on what was
# actually sent.

# ---------------------------------------------------------------------------
# Fixture files
# ---------------------------------------------------------------------------

#' Resolve the path of a fixture file.
#'
#' Works whether the working directory is `tests/testthat` (as it is under
#' `test_check()`) or the package root (as it is under `devtools::test()`).
#'
#' @param name Fixture name, with or without the `.json` suffix.
#' @return Path to the fixture file.
fixture_path <- function(name) {
  if (!grepl("\\.json$", name)) {
    name <- paste0(name, ".json")
  }
  candidates <- c(
    file.path("fixtures", name),
    tryCatch(testthat::test_path("fixtures", name), error = function(e) NULL)
  )
  for (path in candidates) {
    if (file.exists(path)) {
      return(path)
    }
  }
  stop(sprintf("No such fixture: %s", name), call. = FALSE)
}

#' Read a fixture as parsed JSON.
#'
#' Parsed with `simplifyVector = FALSE` so that the FHIR structure is preserved
#' exactly as nested lists.
#'
#' @param name Fixture name, with or without the `.json` suffix.
#' @return The parsed fixture, as a list.
read_fixture <- function(name) {
  jsonlite::fromJSON(fixture_path(name), simplifyVector = FALSE)
}

#' Build an httr2 response from a fixture.
#'
#' Use this directly when a route needs a non-2xx status; otherwise routes can
#' simply name the fixture.
#'
#' @param name Fixture name, with or without the `.json` suffix.
#' @param status HTTP status code.
#' @param url URL to attribute the response to.
#' @return An `httr2_response`.
fixture_response <- function(name,
                             status = 200L,
                             url = "https://tx.example.org/fhir") {
  path <- fixture_path(name)
  httr2::response(
    status_code = status,
    url = url,
    headers = list(`Content-Type` = "application/fhir+json"),
    body = readBin(path, "raw", file.size(path))
  )
}

# ---------------------------------------------------------------------------
# Recorded requests
# ---------------------------------------------------------------------------

fixture_state <- new.env(parent = emptyenv())

reset_fixture_state <- function() {
  fixture_state$requests <- list()
  fixture_state$routes <- list()
  fixture_state$cursor <- list(
    expand = 0L,
    closure_initialize = 0L,
    closure_update = 0L
  )
  invisible(NULL)
}

reset_fixture_state()

#' The method of a request.
#'
#' httr2 only sets `method` explicitly when `req_method()` was used, and
#' otherwise infers it from the presence of a body.
#'
#' @param req An `httr2_request`.
#' @return "GET" or "POST" (or whatever was set explicitly).
request_method <- function(req) {
  if (!is.null(req$method)) {
    return(toupper(req$method))
  }
  if (is.null(req$body)) "GET" else "POST"
}

#' The parsed body of a request.
#'
#' Handles both `req_body_json()` (which holds the body as an R list) and
#' `req_body_raw()` (which holds it as a string or raw vector).
#'
#' @param req An `httr2_request`.
#' @return The body as a list, or NULL when the request has no body.
request_body_data <- function(req) {
  body <- req$body
  if (is.null(body)) {
    return(NULL)
  }
  if (identical(body$type, "json")) {
    return(body$data)
  }
  data <- body$data
  if (is.raw(data)) {
    data <- rawToChar(data)
    Encoding(data) <- "UTF-8"
  }
  if (is.character(data)) {
    return(jsonlite::fromJSON(data, simplifyVector = FALSE))
  }
  data
}

record_request <- function(req) {
  parsed <- httr2::url_parse(req$url)
  record <- list(
    method = request_method(req),
    url = req$url,
    path = parsed$path,
    query = parsed$query,
    headers = req$headers,
    body = request_body_data(req)
  )
  fixture_state$requests <- c(fixture_state$requests, list(record))
  record
}

#' All requests recorded during the most recent `with_fixture_api()` block.
#'
#' @return A list of request records, in the order they were made. Each record
#'   has `method`, `url`, `path`, `query`, `headers` and `body`.
fixture_requests <- function() {
  fixture_state$requests
}

#' The recorded `ValueSet/$expand` requests.
#'
#' @return A list of request records.
expand_requests <- function() {
  Filter(
    function(r) grepl("ValueSet/\\$expand$", r$path),
    fixture_state$requests
  )
}

#' The recorded `$closure` requests.
#'
#' @param type "all", "initialize" or "update".
#' @return A list of request records.
closure_requests <- function(type = c("all", "initialize", "update")) {
  type <- match.arg(type)
  requests <- Filter(
    function(r) grepl("(^|/)\\$closure$", r$path),
    fixture_state$requests
  )
  if (type == "all") {
    return(requests)
  }
  wanted <- type == "update"
  Filter(function(r) has_concept_parameter(r$body) == wanted, requests)
}

#' The values of a query parameter on a recorded request.
#'
#' Repeated parameters (such as `property`) return one element per occurrence,
#' in the order they were sent.
#'
#' @param record A request record.
#' @param name Query parameter name.
#' @return A character vector, empty when the parameter was not sent.
request_query <- function(record, name) {
  values <- record$query[names(record$query) == name]
  if (length(values) == 0L) {
    return(character())
  }
  as.character(unlist(values, use.names = FALSE))
}

#' The `Parameters.parameter` entries of a recorded request body.
#'
#' Nested parameter arrays are flattened, so that a body which wraps its
#' concept parameters in a sub-array is still inspectable.
#'
#' @param record A request record.
#' @return A list of parameter entries.
request_parameters <- function(record) {
  flatten_parameters(record$body$parameter)
}

flatten_parameters <- function(parameters) {
  if (!is.list(parameters)) {
    return(list())
  }
  out <- list()
  for (entry in parameters) {
    if (is.list(entry) && is.null(entry$name)) {
      out <- c(out, flatten_parameters(entry))
    } else {
      out <- c(out, list(entry))
    }
  }
  out
}

has_concept_parameter <- function(body) {
  any(vapply(
    flatten_parameters(body$parameter),
    function(p) identical(p$name, "concept"),
    logical(1)
  ))
}

#' The `name` parameter of a recorded `$closure` request.
#'
#' @param record A request record.
#' @return The closure table name, or NULL when absent.
closure_name <- function(record) {
  for (entry in request_parameters(record)) {
    if (identical(entry$name, "name")) {
      return(entry$valueString)
    }
  }
  NULL
}

#' The codings sent by a recorded `$closure` update request.
#'
#' @param record A request record.
#' @return A list of `valueCoding` objects, in the order they were sent.
closure_concepts <- function(record) {
  concepts <- Filter(
    function(p) identical(p$name, "concept"),
    request_parameters(record)
  )
  lapply(concepts, function(p) p$valueCoding)
}

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

as_route <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  if (inherits(x, "httr2_response")) {
    return(list(x))
  }
  as.list(x)
}

as_route_response <- function(entry) {
  if (inherits(entry, "httr2_response")) {
    return(entry)
  }
  fixture_response(entry)
}

serve_route <- function(route, record) {
  entries <- fixture_state$routes[[route]]
  if (is.null(entries) || length(entries) == 0L) {
    stop(
      sprintf(
        "No fixture configured for the '%s' route (request: %s).",
        route, record$url
      ),
      call. = FALSE
    )
  }

  # Named entries are keyed by the `offset` query parameter, so that paging
  # tests can pin a fixture to a specific page.
  keys <- names(entries)
  if (!is.null(keys) && all(nzchar(keys))) {
    offset <- request_query(record, "offset")
    if (length(offset) != 1L || !(offset %in% keys)) {
      stop(
        sprintf(
          "No '%s' fixture for offset '%s'.",
          route, paste(offset, collapse = ",")
        ),
        call. = FALSE
      )
    }
    return(as_route_response(entries[[offset]]))
  }

  # Otherwise entries are served in request order.
  position <- fixture_state$cursor[[route]] + 1L
  if (position > length(entries)) {
    stop(
      sprintf(
        "Exhausted the '%s' fixtures after %d request(s) (request: %s).",
        route, length(entries), record$url
      ),
      call. = FALSE
    )
  }
  fixture_state$cursor[[route]] <- position
  as_route_response(entries[[position]])
}

fixture_router <- function(req) {
  record <- record_request(req)
  if (grepl("ValueSet/\\$expand$", record$path)) {
    return(serve_route("expand", record))
  }
  if (grepl("(^|/)\\$closure$", record$path)) {
    route <- if (has_concept_parameter(record$body)) {
      "closure_update"
    } else {
      "closure_initialize"
    }
    return(serve_route(route, record))
  }
  stop(sprintf("Unroutable request: %s %s", record$method, record$url),
    call. = FALSE
  )
}

#' Run code with the terminology server answered from fixtures.
#'
#' Resets the request log, installs the mock router, and evaluates `expr`.
#' Afterwards `fixture_requests()`, `expand_requests()` and
#' `closure_requests()` report what was sent.
#'
#' Each route takes either a single fixture name or a vector of names served in
#' request order. An `expand` vector may instead be named by `offset` value, in
#' which case the fixture is chosen by the offset of each request. Any entry may
#' be an `httr2_response` built with `fixture_response()`, which is how a
#' non-2xx response is simulated.
#'
#' @param expr Code to evaluate.
#' @param expand Fixture(s) answering `ValueSet/$expand`.
#' @param closure_initialize Fixture(s) answering the `$closure` initialisation
#'   request (a body with no `concept` parameter).
#' @param closure_update Fixture(s) answering `$closure` update requests (a body
#'   with at least one `concept` parameter).
#' @return The value of `expr`.
#'
#' @examples
#' \dontrun{
#' encoder <- with_fixture_api(
#'   fhir_tx_encoder(scope = "http://example.org/vs", tx_url = "https://tx.example.org/fhir"),
#'   expand = "expand-single-page",
#'   closure_update = "closure-update"
#' )
#' expect_equal(request_query(expand_requests()[[1]], "offset"), "0")
#'
#' # Two pages, chosen by offset.
#' with_fixture_api(
#'   fhir_tx_encoder(scope = "http://example.org/vs", batch_size = 4),
#'   expand = c("0" = "expand-page-1", "4" = "expand-page-2"),
#'   closure_update = c("closure-update-batch-1", "closure-update-batch-2")
#' )
#' }
with_fixture_api <- function(expr,
                             expand = NULL,
                             closure_initialize = "closure-initialize",
                             closure_update = NULL) {
  reset_fixture_state()
  fixture_state$routes <- list(
    expand = as_route(expand),
    closure_initialize = as_route(closure_initialize),
    closure_update = as_route(closure_update)
  )
  httr2::with_mocked_responses(fixture_router, expr)
}
