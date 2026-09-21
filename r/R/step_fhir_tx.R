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

#' Encode a code column with a FHIR terminology server
#'
#' Adds a step to a recipe that replaces a column of terminology codes with the
#' encoding produced by [fhir_tx_encoder()]: one numeric column per feature,
#' named after that feature.
#'
#' The encoder is fitted during `prep()` from `scope` alone - the terminology
#' server defines the feature space, so the training data plays no part in it
#' and nothing about the training data leaks into the encoding. `bake()` is then
#' a lookup, and works for any code within the scope whether or not it appeared
#' in the training data.
#'
#' Because a recipe carries its data as a data frame, `bake()` produces dense
#' columns - one per feature, for every row. A scope of a few hundred concepts
#' is comfortable; a scope of tens of thousands is not, and is much better
#' served by calling [fhir_tx_encoder()] and [transform()] directly, which keeps
#' the result sparse.
#'
#' recipes is a suggested dependency of this package. `step_fhir_tx()` errors if
#' recipes is not installed.
#'
#' @param recipe A recipe object. The step is appended to its existing steps.
#' @param ... One or more tidyselect expressions selecting the code column.
#'   Exactly one column must be selected.
#' @param scope A FHIR ValueSet URI defining the codes to encode, such as a
#'   SNOMED CT implicit value set built from an ECL expression.
#' @param tx_url A FHIR terminology server endpoint. Defaults to the CSIRO
#'   Ontoserver public endpoint.
#' @param subsumption Whether to include subsumption relationships in the
#'   encoding. When `FALSE` the encoding is a plain one-hot encoding.
#' @param properties Property codes to include in the encoding, or `NULL` for
#'   none. A single value of `"*"` requests all properties.
#' @param batch_size The number of codes to send to the terminology server at a
#'   time, both when expanding and when updating the closure table.
#' @param role The role assigned to the columns the step creates.
#' @param trained Whether the step has been prepped. Set by `prep()`; not
#'   normally supplied by the caller.
#' @param skip Whether to skip the step when baking new data. Leave as `FALSE`;
#'   the encoding is needed at prediction time as well as at training time.
#' @param id A unique identifier for the step.
#' @return An updated recipe with the new step appended.
#' @seealso [fhir_tx_encoder()] for the functional equivalent, which returns a
#'   sparse matrix.
#' @examples
#' \dontrun{
#' library(recipes)
#'
#' data <- data.frame(
#'   code = c("399981008", "363346000"),
#'   outcome = c(0, 1)
#' )
#'
#' rec <- recipe(outcome ~ code, data = data) |>
#'   step_fhir_tx(
#'     code,
#'     # Ancestors of the SNOMED CT concept "Malignant neoplastic disease".
#'     scope = "http://snomed.info/sct?fhir_vs=ecl/(%3E%3E%20363346000)"
#'   ) |>
#'   prep()
#'
#' # One column per code in the scope, plus the untouched outcome.
#' bake(rec, new_data = data)
#' }
#' @export
step_fhir_tx <- function(recipe,
                         ...,
                         scope,
                         tx_url = "https://tx.ontoserver.csiro.au/fhir",
                         subsumption = TRUE,
                         properties = NULL,
                         batch_size = 50000,
                         role = "predictor",
                         trained = FALSE,
                         skip = FALSE,
                         id = recipes::rand_id("fhir_tx")) {
  if (!recipes_installed()) {
    stop(
      "The recipes package must be installed to use step_fhir_tx().",
      call. = FALSE
    )
  }
  recipes::add_step(
    recipe,
    step_fhir_tx_new(
      terms = recipes::ellipse_check(...),
      scope = scope,
      tx_url = tx_url,
      subsumption = subsumption,
      properties = properties,
      batch_size = batch_size,
      role = role,
      trained = trained,
      encoder = NULL,
      columns = NULL,
      skip = skip,
      id = id
    )
  )
}

#' Report whether recipes is available.
#'
#' Factored out so that tests can simulate recipes being absent.
#'
#' @return `TRUE` when the recipes package can be loaded.
#' @noRd
recipes_installed <- function() {
  requireNamespace("recipes", quietly = TRUE)
}

#' Construct a `step_fhir_tx` object.
#'
#' @param terms Quosures selecting the code column.
#' @param scope,tx_url,subsumption,properties,batch_size Encoder parameters.
#' @param role,trained,skip,id Standard recipes step fields.
#' @param encoder The fitted encoder, or `NULL` before `prep()`.
#' @param columns The selected column name, or `NULL` before `prep()`.
#' @return A `step_fhir_tx` object.
#' @noRd
step_fhir_tx_new <- function(terms, scope, tx_url, subsumption, properties,
                             batch_size, role, trained, encoder, columns,
                             skip, id) {
  recipes::step(
    subclass = "fhir_tx",
    terms = terms,
    scope = scope,
    tx_url = tx_url,
    subsumption = subsumption,
    properties = properties,
    batch_size = batch_size,
    role = role,
    trained = trained,
    encoder = encoder,
    columns = columns,
    skip = skip,
    id = id
  )
}

#' Fit the encoder for a `step_fhir_tx` step.
#'
#' @param x A `step_fhir_tx` object.
#' @param training The training data, used only to resolve the column
#'   selection.
#' @param info A tibble describing the variables available to the step.
#' @param ... Unused.
#' @return A trained `step_fhir_tx` object carrying the fitted encoder.
#' @exportS3Method recipes::prep
#' @noRd
prep.step_fhir_tx <- function(x, training, info = NULL, ...) {
  col_names <- unname(recipes::recipes_eval_select(x$terms, training, info))
  if (length(col_names) != 1L) {
    detail <- if (length(col_names) == 0L) {
      "none were selected."
    } else {
      sprintf(
        "%d were selected: %s.",
        length(col_names), paste(col_names, collapse = ", ")
      )
    }
    stop(
      "step_fhir_tx() requires exactly one code column, but ", detail,
      call. = FALSE
    )
  }

  # The scope, not the training data, defines the feature space.
  encoder <- fhir_tx_encoder(
    scope = x$scope,
    tx_url = x$tx_url,
    subsumption = x$subsumption,
    properties = x$properties,
    batch_size = x$batch_size
  )

  step_fhir_tx_new(
    terms = x$terms,
    scope = x$scope,
    tx_url = x$tx_url,
    subsumption = x$subsumption,
    properties = x$properties,
    batch_size = x$batch_size,
    role = x$role,
    trained = TRUE,
    encoder = encoder,
    columns = col_names,
    skip = x$skip,
    id = x$id
  )
}

#' Apply the encoding of a `step_fhir_tx` step.
#'
#' @param object A trained `step_fhir_tx` object.
#' @param new_data A tibble of data to encode.
#' @param ... Unused.
#' @return `new_data` with the code column replaced by one numeric column per
#'   feature of the encoder.
#' @exportS3Method recipes::bake
#' @noRd
bake.step_fhir_tx <- function(object, new_data, ...) {
  col_name <- object$columns
  recipes::check_new_data(col_name, object, new_data)

  # A factor column carries its codes as labels, so coerce before encoding.
  codes <- as.character(new_data[[col_name]])
  dense <- as.matrix(
    transform(object$encoder, matrix(codes, ncol = 1))
  )
  colnames(dense) <- object$encoder$feature_names

  retained <- setdiff(names(new_data), col_name)
  tibble::as_tibble(c(new_data[retained], tibble::as_tibble(dense)))
}

#' Summarise a `step_fhir_tx` step.
#'
#' @param x A `step_fhir_tx` object.
#' @param width The width to wrap the column listing at.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @exportS3Method base::print
#' @noRd
print.step_fhir_tx <- function(x, width = max(20, options()$width - 30), ...) {
  recipes::print_step(
    x$columns, x$terms, x$trained, "FHIR terminology encoding for ", width
  )
  invisible(x)
}

#' Report the parameters of a `step_fhir_tx` step.
#'
#' @param x A `step_fhir_tx` object.
#' @param ... Unused.
#' @return A one-row tibble of the step's parameters. `properties` is a list
#'   column, because a step may request any number of properties.
#' @exportS3Method recipes::tidy
#' @noRd
tidy.step_fhir_tx <- function(x, ...) {
  terms <- if (recipes::is_trained(x)) {
    x$columns
  } else {
    recipes::sel2char(x$terms)
  }
  tibble::tibble(
    terms = terms,
    scope = x$scope,
    tx_url = x$tx_url,
    subsumption = x$subsumption,
    properties = list(x$properties),
    batch_size = x$batch_size,
    id = x$id
  )
}

#' Report the packages a `step_fhir_tx` step needs.
#'
#' @param x A `step_fhir_tx` object.
#' @param ... Unused.
#' @return A character vector of package names.
#' @exportS3Method recipes::required_pkgs
#' @noRd
required_pkgs.step_fhir_tx <- function(x, ...) {
  "fhirtxencoder"
}
