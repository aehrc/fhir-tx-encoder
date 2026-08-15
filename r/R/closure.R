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

# Wrappers over the FHIR `$closure` operation. The closure table lives on the
# server; these functions only return the new pairs from each update, so the
# whole table never has to be held in memory.

#' Create a server-side transitive closure table.
#'
#' @param tx_url A FHIR terminology server endpoint.
#' @return The name of the new closure table, a UUID with its hyphens removed.
#' @noRd
closure_initialize <- function(tx_url) {
  name <- new_uuid_hex()
  closure_request(tx_url, list(
    resourceType = "Parameters",
    parameter = list(list(name = "name", valueString = name))
  ))
  name
}

#' Add a batch of codings to a closure table.
#'
#' The codings are reduced to `system`, `version` and `code` - the server needs
#' nothing else, and sending displays or property extensions back would be
#' wasteful.
#'
#' @param tx_url A FHIR terminology server endpoint.
#' @param name The closure table name returned by `closure_initialize()`.
#' @param codings A list of FHIR Coding objects.
#' @return A list with `source` and `target` character vectors of equal length,
#'   holding the new subsumption pairs: the target of each pair subsumes its
#'   source.
#' @noRd
closure_update <- function(tx_url, name, codings) {
  concepts <- lapply(
    Filter(Negate(is.null), codings),
    function(coding) list(name = "concept", valueCoding = minimal_coding(coding))
  )
  concept_map <- closure_request(tx_url, list(
    resourceType = "Parameters",
    parameter = c(list(list(name = "name", valueString = name)), concepts)
  ))
  subsumption_pairs(concept_map)
}

#' A coding reduced to the fields the closure operation needs.
#'
#' @param coding A FHIR Coding, as a list.
#' @return The coding with only its `system`, `version` and `code` fields, in
#'   that order, omitting any that are absent.
#' @noRd
minimal_coding <- function(coding) {
  coding[intersect(c("system", "version", "code"), names(coding))]
}

#' The subsumption pairs of a closure update response.
#'
#' A pair is `(element.code, target.code)` for every target whose equivalence
#' is `subsumes`; in FHIR R4 that means the target subsumes the element. All
#' other equivalences are ignored, as is a response with no `group`.
#'
#' @param concept_map A ConceptMap, as a list.
#' @return A list with `source` and `target` character vectors.
#' @noRd
subsumption_pairs <- function(concept_map) {
  pairs <- list()
  for (group in concept_map$group) {
    for (element in group$element) {
      for (target in element$target) {
        if (identical(target$equivalence, "subsumes")) {
          pairs[[length(pairs) + 1L]] <- c(element$code, target$code)
        }
      }
    }
  }
  if (length(pairs) == 0L) {
    return(list(source = character(), target = character()))
  }
  pairs <- do.call(rbind, pairs)
  list(source = pairs[, 1], target = pairs[, 2])
}

#' Post a Parameters body to the `$closure` endpoint.
#'
#' @param tx_url A FHIR terminology server endpoint.
#' @param body The Parameters resource, as a list.
#' @return The parsed response body.
#' @noRd
closure_request <- function(tx_url, body) {
  request <- httr2::req_body_json(
    httr2::req_headers(
      httr2::request(paste0(tx_url, "/$closure")),
      Accept = "application/fhir+json"
    ),
    body,
    type = "application/fhir+json"
  )
  httr2::resp_body_json(httr2::req_perform(request), simplifyVector = FALSE)
}

#' Generate a version 4 UUID with its hyphens removed.
#'
#' Matches the closure table names the Python implementation generates with
#' `uuid.uuid4().hex`.
#'
#' The bytes are drawn from a stream reseeded from the clock and the process id
#' (`set.seed(NULL)`), and the caller's `.Random.seed` is restored afterwards.
#' Drawing from the caller's own stream would be wrong twice over: a caller who
#' had set a seed for a reproducible workflow would get a predictable table
#' name - so two such callers on the same public server would share a closure
#' table and corrupt each other's encodings - and constructing an encoder would
#' silently advance their random stream.
#'
#' @return A 32 character lower-case hexadecimal string.
#' @noRd
new_uuid_hex <- function() {
  seeded <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  if (seeded) {
    saved <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
  }
  on.exit(
    if (seeded) {
      assign(".Random.seed", saved, envir = globalenv())
    } else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      # The caller had not used the generator at all, so leave no trace of it.
      rm(".Random.seed", envir = globalenv())
    },
    add = TRUE
  )

  set.seed(NULL)
  bytes <- sample.int(256L, 16L, replace = TRUE) - 1L
  # Byte 7 carries the version nibble, byte 9 the variant bits.
  bytes[7] <- bitwOr(bitwAnd(bytes[7], 0x0f), 0x40)
  bytes[9] <- bitwOr(bitwAnd(bytes[9], 0x3f), 0x80)
  paste(sprintf("%02x", bytes), collapse = "")
}
