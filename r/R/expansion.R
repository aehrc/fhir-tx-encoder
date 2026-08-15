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

# The R5 pre-adoption extension through which current servers convey expansion
# properties in an R4 response.
EXPANSION_PROPERTY_PREADOPT <- paste0(
  "http://hl7.org/fhir/5.0/StructureDefinition/",
  "extension-ValueSet.expansion.contains.property"
)

# The subsumption encoding already carries the hierarchy, so these properties
# would only duplicate it.
IGNORED_PROPERTIES <- c("parent", "child")

#' Expand a value set scope, one page at a time.
#'
#' Issues `GET <tx_url>/ValueSet/$expand` repeatedly, advancing the offset by
#' `batch_size` after each page and continuing while the offset does not exceed
#' the total the server reports in `expansion.total`.
#'
#' Progress is reported with `message()`, so it can be silenced with
#' `suppressMessages()`.
#'
#' Errors when the expansion yields no codes, and propagates the httr2 error
#' carrying the status code when the server responds with a non-2xx status.
#'
#' @param scope A FHIR ValueSet URI defining the codes to encode.
#' @param tx_url A FHIR terminology server endpoint.
#' @param properties Character vector of property codes to request, or NULL for
#'   none. Each code is sent as a separate `property` query parameter.
#' @param batch_size Number of codes to request per page.
#' @return A list with `codes` (character), `displays` (character, `NA` where
#'   the server sent no display), `coding_batches` (a list holding the raw
#'   codings of each page, retained for the closure updates) and `properties`
#'   (one named list of property values per concept, in the same order as
#'   `codes`).
#' @noRd
expand_scope <- function(scope, tx_url, properties, batch_size) {
  message("Expanding value set: ", scope)

  batch_size <- as.integer(batch_size)
  offset <- 0L
  total <- 0L
  codes <- character()
  displays <- character()
  coding_batches <- list()
  concept_properties <- list()

  while (offset <= total) {
    response <- httr2::req_perform(
      expansion_request(scope, tx_url, properties, batch_size, offset)
    )
    expansion <- httr2::resp_body_json(
      response,
      simplifyVector = FALSE
    )$expansion
    total <- as.integer(expansion$total)

    # An exhausted or empty expansion has no `contains` element at all.
    codings <- expansion$contains
    if (is.null(codings)) {
      codings <- list()
    }
    message(sprintf(
      "Expanding (%d items, offset %d, total %d)",
      length(codings), offset, total
    ))

    coding_batches <- c(coding_batches, list(codings))
    codes <- c(codes, vapply(codings, function(x) x$code, character(1)))
    displays <- c(displays, vapply(codings, coding_display, character(1)))
    # Properties are extracted whether or not they were requested: a server
    # that sends none leaves an empty list per concept, and the encoder only
    # widens the encoding when properties were asked for.
    concept_properties <- c(
      concept_properties, lapply(codings, properties_to_list)
    )

    offset <- offset + batch_size
  }

  if (length(codes) == 0L) {
    stop("Value set expansion is empty: ", scope, call. = FALSE)
  }
  message("Expansion complete")

  list(
    codes = codes,
    displays = displays,
    coding_batches = coding_batches,
    properties = concept_properties
  )
}

#' Build the request for a single page of an expansion.
#'
#' @param scope A FHIR ValueSet URI.
#' @param tx_url A FHIR terminology server endpoint.
#' @param properties Character vector of property codes, or NULL.
#' @param batch_size Number of codes to request.
#' @param offset Offset of the first code to request.
#' @return An `httr2_request`.
#' @noRd
expansion_request <- function(scope, tx_url, properties, batch_size, offset) {
  request <- httr2::req_headers(
    httr2::req_url_query(
      httr2::request(paste0(tx_url, "/ValueSet/$expand")),
      url = scope,
      count = as.character(batch_size),
      offset = as.character(offset)
    ),
    Accept = "application/fhir+json"
  )
  if (is.null(properties)) {
    return(request)
  }
  # Each requested property code is sent as its own `property` parameter.
  httr2::req_url_query(request, property = properties, .multi = "explode")
}

#' The properties of a coding, flattened into a named list.
#'
#' Two response formats are supported and produce identical results: the native
#' R4 `expansion.contains.property` elements, and the R5 pre-adoption extension
#' that current servers use. A coding carrying native properties is read that
#' way; otherwise its extensions are examined.
#'
#' Subproperties are flattened onto their parent's key, recursively, so a
#' subproperty `b` of property `a` becomes the key `a.b`. The property codes
#' `parent` and `child` are excluded at every level, a property with no value of
#' its own contributes no entry (though its subproperties still do), and
#' malformed entries are skipped rather than raising an error.
#'
#' For example, a coding whose only property is a SNOMED CT role group
#' (`609096000`) holding an associated morphology subproperty (`116676008`) of
#' "Neoplasm" (`108369006`) yields
#' `list("609096000.116676008" = "108369006")`, in either format.
#'
#' @param coding A FHIR Coding, as a list.
#' @return A named list of property values, empty when the coding carries none.
#'   Values are length-one character or numeric vectors.
#' @noRd
properties_to_list <- function(coding) {
  if (!is.null(coding$property)) {
    return(native_properties(coding$property))
  }
  if (!is.null(coding$extension)) {
    return(extension_properties(coding$extension))
  }
  list()
}

#' The properties of a coding expressed as native property elements.
#'
#' @param elements The `property` elements of a coding.
#' @return A named list of property values.
#' @noRd
native_properties <- function(elements) {
  result <- list()
  for (element in elements) {
    code <- element$code
    if (!included_property(code)) {
      next
    }
    # The value is read from the property element itself. The Python
    # implementation reads the `value[x]` key off the coding instead, which is
    # a defect: a coding has no such key, so its native branch never finds a
    # value. The intended behaviour is followed here.
    result <- set_property(result, code, value_of(element))
    result <- merge_properties(result, subproperties(element, code))
  }
  result
}

#' The properties of a coding expressed as R5 pre-adoption extensions.
#'
#' @param extensions The `extension` elements of a coding. Extensions with any
#'   other URL are ignored.
#' @return A named list of property values.
#' @noRd
extension_properties <- function(extensions) {
  result <- list()
  for (extension in extensions) {
    if (!identical(extension$url, EXPANSION_PROPERTY_PREADOPT)) {
      next
    }
    code <- sub_extension_value(extension, "code")
    if (!included_property(code)) {
      next
    }
    value <- sub_extension_value(extension, "value")
    result <- set_property(result, code, value)
    result <- merge_properties(result, subproperties(extension, code))
  }
  result
}

#' The flattened subproperties of a property element or extension.
#'
#' Both formats carry subproperties the same way: `subproperty` extensions with
#' `code` and `value` sub-extensions, nested to any depth.
#'
#' @param element A property element or property extension.
#' @param key The flattened key of the element the subproperties belong to.
#' @return A named list of property values, keyed by `key` followed by the
#'   subproperty code.
#' @noRd
subproperties <- function(element, key) {
  result <- list()
  for (extension in element$extension) {
    if (!identical(extension$url, "subproperty")) {
      next
    }
    code <- sub_extension_value(extension, "code")
    if (!included_property(code)) {
      next
    }
    subkey <- paste0(key, ".", code)
    value <- sub_extension_value(extension, "value")
    result <- set_property(result, subkey, value)
    result <- merge_properties(result, subproperties(extension, subkey))
  }
  result
}

#' Whether a property code is usable and wanted.
#'
#' @param code The property code, which is NULL when the entry is malformed.
#' @return `TRUE` when the code is a single string other than `parent` or
#'   `child`.
#' @noRd
included_property <- function(code) {
  is.character(code) && length(code) == 1L && !(code %in% IGNORED_PROPERTIES)
}

#' The `value[x]` of a property element or extension.
#'
#' @param element A property element, or a `value` sub-extension.
#' @return The value, or NULL when there is none or it is composite (a
#'   `valueCoding`, say, has no scalar form and so cannot name a feature).
#' @noRd
value_of <- function(element) {
  keys <- names(element)
  for (key in keys[startsWith(keys, "value")]) {
    value <- element[[key]]
    if (is.atomic(value) && length(value) == 1L) {
      return(value)
    }
  }
  NULL
}

#' The value of the sub-extension with a given URL.
#'
#' @param element A property element or extension holding sub-extensions.
#' @param url The URL of the sub-extension to read, `"code"` or `"value"`.
#' @return The sub-extension's value, or NULL when it is absent or carries no
#'   value.
#' @noRd
sub_extension_value <- function(element, url) {
  for (extension in element$extension) {
    if (identical(extension$url, url)) {
      return(value_of(extension))
    }
  }
  NULL
}

#' Record a property value, unless there is none.
#'
#' @param result The named list being built.
#' @param key The flattened property key.
#' @param value The property value, or NULL.
#' @return `result`, with `key` set when `value` is not NULL.
#' @noRd
set_property <- function(result, key, value) {
  if (is.null(value)) {
    return(result)
  }
  result[[key]] <- value
  result
}

#' Merge property values into the list being built.
#'
#' A repeated key takes the later value, which is how the Python
#' implementation's dictionary behaves.
#'
#' @param result The named list being built.
#' @param additions A named list of property values.
#' @return The merged named list.
#' @noRd
merge_properties <- function(result, additions) {
  if (length(additions) > 0L) {
    result[names(additions)] <- additions
  }
  result
}

#' Widen per-concept property lists into a sparse matrix.
#'
#' Replicates scikit-learn's `DictVectorizer` with its default `sort = True`,
#' so that the R and Python encoders produce the same columns in the same
#' order:
#'
#' - a character value becomes an indicator feature named `key=value`, whose
#'   cell holds 1;
#' - a numeric (or logical) value becomes a feature named `key`, whose cell
#'   holds the value itself;
#' - feature names are sorted by byte order rather than by the user's locale
#'   collation, matching Python's string ordering;
#' - a concept without a given feature has 0 in that cell.
#'
#' For example, `list(list(a = "x"), list(a = "y", n = 2.5))` gives the feature
#' names `c("a=x", "a=y", "n")` and the rows `(1, 0, 0)` and `(0, 1, 2.5)`.
#'
#' @param property_lists One named list of property values per concept, in
#'   concept order, as returned by `properties_to_list()`.
#' @return A list with `matrix`, a `Matrix::dgCMatrix` with one row per concept
#'   and one column per feature, and `feature_names`, the column names in
#'   order.
#' @noRd
vectorize_properties <- function(property_lists) {
  entries <- unlist(property_lists, recursive = FALSE, use.names = TRUE)
  rows <- rep.int(seq_along(property_lists), lengths(property_lists))
  keys <- names(entries)
  features <- vapply(
    seq_along(entries),
    function(position) feature_name(keys[[position]], entries[[position]]),
    character(1)
  )
  values <- vapply(entries, feature_value, numeric(1), USE.NAMES = FALSE)

  # `method = "radix"` sorts by byte order in the C locale, as Python does;
  # the default method would sort by the user's locale collation.
  feature_names <- sort(unique(features), method = "radix")
  list(
    matrix = Matrix::sparseMatrix(
      i = rows,
      j = match(features, feature_names),
      x = values,
      dims = c(length(property_lists), length(feature_names))
    ),
    feature_names = feature_names
  )
}

#' The feature a property value contributes.
#'
#' @param key The flattened property key.
#' @param value The property value.
#' @return The feature name: the key alone for a numeric value, `key=value`
#'   otherwise.
#' @noRd
feature_name <- function(key, value) {
  if (is_numeric_value(value)) key else paste0(key, "=", value)
}

#' The cell a property value contributes.
#'
#' @param value The property value.
#' @return The value itself when it is numeric, otherwise 1 for the indicator.
#' @noRd
feature_value <- function(value) {
  if (is_numeric_value(value)) as.numeric(value) else 1
}

#' Whether a property value is carried as a number rather than an indicator.
#'
#' Logical values count as numeric, matching Python, where a boolean is a
#' number.
#'
#' @param value The property value.
#' @return `TRUE` when the value is numeric or logical.
#' @noRd
is_numeric_value <- function(value) {
  is.numeric(value) || is.logical(value)
}

#' The display of a coding, or NA when it has none.
#'
#' @param coding A FHIR Coding, as a list.
#' @return A length-one character vector.
#' @noRd
coding_display <- function(coding) {
  if (is.null(coding$display)) NA_character_ else coding$display
}
