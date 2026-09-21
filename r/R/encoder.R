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

#' Build a FHIR terminology encoder
#'
#' Expands a FHIR ValueSet URI against a terminology server and builds a sparse
#' encoding of every code in the resulting scope. Each row of the encoding
#' carries a 1 for the code itself and a 1 for every code in the scope that
#' subsumes it, so that codes which are ontologically close share features.
#'
#' The encoder is fitted eagerly at construction: all terminology server
#' requests happen here, and `transform()` afterwards is a pure lookup.
#'
#' When `properties` are requested, the encoding is widened with one column per
#' distinct property value observed in the scope. A property value contributes
#' an indicator column named `property=value`, a numeric property value
#' contributes a single column named `property` carrying the number, and nested
#' subproperties are flattened onto their parent, as in
#' `609096000.116676008=108369006`. The property columns follow the code
#' columns, in byte order of their names.
#'
#' Progress is reported with `message()` and can be silenced with
#' `suppressMessages()`.
#'
#' @param scope A FHIR ValueSet URI defining the codes to encode, such as a
#'   SNOMED CT implicit value set built from an ECL expression - see the
#'   examples.
#' @param tx_url A FHIR terminology server endpoint. Defaults to the CSIRO
#'   Ontoserver public endpoint.
#' @param subsumption Whether to include subsumption relationships in the
#'   encoding. When `FALSE` the encoding is a plain one-hot encoding.
#' @param properties Property codes to include in the encoding, or `NULL` for
#'   none. A single value of `"*"` requests all properties.
#' @param batch_size The number of codes to send to the terminology server at a
#'   time, both when expanding and when updating the closure table.
#' @return An object of class `fhir_tx_encoder`: a list with `codes` (the codes
#'   in expansion order), `displays` (their display terms, `NA` where the server
#'   sent none), `feature_names` (the name of each column of the encoding),
#'   `encoded` (a `Matrix::dgCMatrix` with one row per code), `index` (a named
#'   integer vector mapping each code to its row and column) and `scope`.
#' @examples
#' \dontrun{
#' # Ancestors of the SNOMED CT concept "Malignant neoplastic disease".
#' encoder <- fhir_tx_encoder(
#'   scope = "http://snomed.info/sct?fhir_vs=ecl/(%3E%3E%20363346000)"
#' )
#' encoder
#'
#' # Encode two concepts.
#' transform(encoder, matrix(c("399981008", "363346000"), ncol = 1))
#' }
#' @export
fhir_tx_encoder <- function(scope,
                            tx_url = "https://tx.ontoserver.csiro.au/fhir",
                            subsumption = TRUE,
                            properties = NULL,
                            batch_size = 50000) {
  expansion <- expand_scope(scope, tx_url, properties, batch_size)
  codes <- expansion$codes
  size <- length(codes)

  # The base encoding is the identity: every code carries itself. It is built
  # from triplets rather than by mutating a sparse matrix, which would copy the
  # whole matrix on every update.
  message(sprintf("Generating one-hot encoding... (%d, %d)", size, size))
  rows <- seq_len(size)
  columns <- seq_len(size)

  # The index maps a code to its row and column, and is how a closure pair is
  # turned into a cell.
  message(sprintf("Creating index... %d items", size))
  index <- stats::setNames(seq_len(size), codes)

  if (subsumption) {
    message("Applying transitive closure...")
    cells <- closure_cells(expansion$coding_batches, tx_url, index)
    rows <- c(rows, cells$rows)
    columns <- c(columns, cells$columns)
  }

  # A cell set by more than one pair must still hold 1, not a count.
  distinct <- !duplicated(cbind(rows, columns))
  encoded <- Matrix::sparseMatrix(
    i = rows[distinct], j = columns[distinct], x = 1,
    dims = c(size, size)
  )
  if (subsumption) {
    message(sprintf(
      "Subsumption encoding complete: (%d, %d)", nrow(encoded), ncol(encoded)
    ))
  }

  # The property features are appended after the code columns, so the first n
  # columns of a widened encoding are still the codes.
  feature_names <- codes
  if (!is.null(properties)) {
    vectorized <- vectorize_properties(expansion$properties)
    encoded <- cbind(encoded, vectorized$matrix)
    feature_names <- c(feature_names, vectorized$feature_names)
    message(sprintf(
      "Encoding properties... (%d, %d)", nrow(encoded), ncol(encoded)
    ))
  }

  structure(
    list(
      codes = codes,
      displays = expansion$displays,
      feature_names = feature_names,
      encoded = encoded,
      index = index,
      scope = scope
    ),
    class = "fhir_tx_encoder"
  )
}

#' Collect the matrix cells implied by the transitive closure of a scope.
#'
#' Runs one closure update per expansion batch, so that the server is never
#' sent more codes at once than the caller asked for.
#'
#' @param coding_batches The raw codings of each expansion page.
#' @param tx_url A FHIR terminology server endpoint.
#' @param index The code to position index.
#' @return A list with `rows` and `columns` integer vectors.
#' @noRd
closure_cells <- function(coding_batches, tx_url, index) {
  name <- closure_initialize(tx_url)
  batch_count <- length(coding_batches)
  rows <- integer()
  columns <- integer()

  for (position in seq_len(batch_count)) {
    batch <- coding_batches[[position]]
    pairs <- closure_update(tx_url, name, batch)
    # The target of a pair subsumes its source, so the cell to set is
    # [row(source), col(target)].
    rows <- c(rows, unname(index[pairs$source]))
    columns <- c(columns, unname(index[pairs$target]))
    message(sprintf(
      "Batch %d of %d, %d items... %d pairs added",
      position, batch_count, length(batch), length(pairs$source)
    ))
  }

  list(rows = rows, columns = columns)
}

#' Encode a matrix of codes
#'
#' Looks up the encoding of every code in `X`. The columns of `X` are encoded
#' independently and bound side by side, so a two-column input over an encoder
#' with `f` features produces `2 * f` columns.
#'
#' @param _data An object of class `fhir_tx_encoder`.
#' @param X A two-dimensional character matrix of codes, with one row per
#'   observation and one column per code variable.
#' @param ... Unused, present for compatibility with the `transform()` generic.
#' @return A `Matrix::dgCMatrix` with `nrow(X)` rows and
#'   `ncol(X) * length(_data$feature_names)` columns.
#' @examples
#' \dontrun{
#' encoder <- fhir_tx_encoder(
#'   scope = "http://snomed.info/sct?fhir_vs=ecl/(%3E%3E%20363346000)"
#' )
#' transform(encoder, matrix(c("399981008", "363346000"), ncol = 1))
#' }
#' @export
transform.fhir_tx_encoder <- function(`_data`, X, ...) {
  if (!is.matrix(X) || !is.character(X)) {
    stop("X must be a two-dimensional character matrix", call. = FALSE)
  }
  encoded <- lapply(
    seq_len(ncol(X)),
    function(column) transform_column(`_data`, X[, column])
  )
  do.call(cbind, encoded)
}

#' Encode a single column of codes.
#'
#' @param encoder An object of class `fhir_tx_encoder`.
#' @param codes A character vector of codes.
#' @return A `Matrix::dgCMatrix` with one row per code.
#' @noRd
transform_column <- function(encoder, codes) {
  positions <- unname(encoder$index[codes])
  unknown <- is.na(positions)
  if (any(unknown)) {
    stop("Encountered code not in scope: ", codes[unknown][[1]], call. = FALSE)
  }
  encoder$encoded[positions, , drop = FALSE]
}

#' Summarise a FHIR terminology encoder
#'
#' @param x An object of class `fhir_tx_encoder`.
#' @param ... Unused, present for compatibility with the `print()` generic.
#' @return `x`, invisibly.
#' @examples
#' \dontrun{
#' encoder <- fhir_tx_encoder(
#'   scope = "http://snomed.info/sct?fhir_vs=ecl/(%3E%3E%20363346000)"
#' )
#' print(encoder)
#' }
#' @export
print.fhir_tx_encoder <- function(x, ...) {
  cat(sprintf(
    "<fhir_tx_encoder> %s - %d codes, %d features\n",
    x$scope, length(x$codes), length(x$feature_names)
  ))
  invisible(x)
}
