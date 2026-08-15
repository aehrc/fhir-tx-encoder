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

# Property extraction from both response formats, and the
# DictVectorizer-equivalent widening of the encoding.

test_scope <- "http://snomed.info/sct?fhir_vs=ecl/(%3E%3E%20363346000)"
test_tx_url <- "https://tx.example.org/fhir"

preadopt_url <- paste0(
  "http://hl7.org/fhir/5.0/StructureDefinition/",
  "extension-ValueSet.expansion.contains.property"
)

expected_codes <- c(
  "404684003", "64572001", "363346000", "399981008", "55342001", "138875005"
)

# The property fixtures hold the same five concepts in the same order, one
# expressed natively and one through the R5 pre-adoption extension.
fixture_properties <- function(name) {
  lapply(read_fixture(name)$expansion$contains, properties_to_list)
}

# What both property fixtures must yield, concept by concept. Documented in
# fixtures/README.md.
expected_fixture_properties <- list(
  # A role group with two subproperties: the group itself has no value, so it
  # contributes no feature of its own.
  list(
    "609096000.116676008" = "1240414004",
    "609096000.363698007" = "39937001"
  ),
  # A subproperty nested inside a subproperty.
  list(
    "609096000.116676008" = "108369006",
    "609096000.116676008.363698007" = "39937001"
  ),
  # A flat property, alongside `parent` and `child` properties.
  list("116676008" = "108369006"),
  # A numeric property.
  list(severityScore = 2.5),
  # A property with no code, plus a `parent` property: nothing survives.
  list()
)

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

test_that("native contains.property elements yield the flattened property list", {
  # Arrange / Act
  properties <- fixture_properties("expand-properties-native")

  # Assert - one list per concept, in expansion order.
  expect_length(properties, 5L)
  expect_equal(properties, expected_fixture_properties)
})

test_that("the R5 pre-adoption extension yields exactly the same property list", {
  # Arrange / Act - the two fixtures differ only in how the properties are
  # expressed, so extraction must not be able to tell them apart.
  native <- fixture_properties("expand-properties-native")
  preadopt <- fixture_properties("expand-properties-preadopt")

  # Assert
  expect_equal(preadopt, expected_fixture_properties)
  expect_identical(preadopt, native)
})

test_that("subproperties flatten recursively onto their parent's code", {
  # Arrange - a property with a value, a subproperty and a sub-subproperty.
  coding <- list(
    code = "363346000",
    property = list(list(
      code = "a",
      valueCode = "1",
      extension = list(list(
        url = "subproperty",
        extension = list(
          list(url = "code", valueCode = "b"),
          list(url = "value", valueCode = "2"),
          list(
            url = "subproperty",
            extension = list(
              list(url = "code", valueCode = "c"),
              list(url = "value", valueCode = "3")
            )
          )
        )
      ))
    ))
  )

  # Act
  result <- properties_to_list(coding)

  # Assert - each level appends `.code` to the key of the level above.
  expect_equal(result, list(a = "1", a.b = "2", a.b.c = "3"))
})

test_that("a property with subproperties but no value of its own contributes no feature", {
  # Arrange - the SNOMED CT role group case: `609096000` carries only
  # subproperties.
  coding <- read_fixture("expand-properties-native")$expansion$contains[[1]]

  # Act
  result <- properties_to_list(coding)

  # Assert - no bare `609096000` key, only the two subproperty keys.
  expect_false("609096000" %in% names(result))
  expect_equal(names(result), c("609096000.116676008", "609096000.363698007"))
})

test_that("parent and child properties are excluded at every level", {
  # Arrange - `parent` and `child` at the top level and beneath a role group.
  coding <- list(
    property = list(
      list(code = "parent", valueCode = "64572001"),
      list(code = "child", valueCode = "363346000"),
      list(
        code = "609096000",
        extension = list(
          list(
            url = "subproperty",
            extension = list(
              list(url = "code", valueCode = "parent"),
              list(url = "value", valueCode = "64572001")
            )
          ),
          list(
            url = "subproperty",
            extension = list(
              list(url = "code", valueCode = "116676008"),
              list(url = "value", valueCode = "108369006")
            )
          )
        )
      )
    )
  )

  # Act
  result <- properties_to_list(coding)

  # Assert - subsumption already carries parent/child, so only the associated
  # morphology subproperty survives.
  expect_equal(result, list("609096000.116676008" = "108369006"))
})

test_that("a concept with no properties yields an empty list", {
  # Arrange / Act - a coding with neither `property` nor `extension`.
  result <- properties_to_list(
    list(system = "http://snomed.info/sct", code = "138875005")
  )

  # Assert
  expect_equal(result, list())
})

test_that("extensions that are not property extensions are ignored", {
  # Arrange / Act - a coding carrying an unrelated extension only.
  result <- properties_to_list(list(
    code = "138875005",
    extension = list(list(url = "http://example.org/ext", valueCode = "x"))
  ))

  # Assert
  expect_equal(result, list())
})

# ---------------------------------------------------------------------------
# Malformed entries
# ---------------------------------------------------------------------------

test_that("a native property with no code is skipped without failing", {
  # Arrange - the malformed entry sits before a well-formed one, so a skip
  # (rather than an abort) is observable.
  coding <- list(property = list(
    list(valueCode = "108369006"),
    list(code = "116676008", valueCode = "108369006")
  ))

  # Act / Assert
  expect_equal(
    properties_to_list(coding), list("116676008" = "108369006")
  )
})

test_that("a property extension with no code sub-extension is skipped", {
  # Arrange
  coding <- list(extension = list(
    list(
      url = preadopt_url,
      extension = list(list(url = "value", valueCode = "108369006"))
    ),
    list(
      url = preadopt_url,
      extension = list(
        list(url = "code", valueCode = "116676008"),
        list(url = "value", valueCode = "108369006")
      )
    )
  ))

  # Act / Assert
  expect_equal(
    properties_to_list(coding), list("116676008" = "108369006")
  )
})

test_that("a value extension with no value[x] contributes no feature", {
  # Arrange - a `value` sub-extension that carries no value at all, both at
  # the property level and at the subproperty level.
  coding <- list(extension = list(list(
    url = preadopt_url,
    extension = list(
      list(url = "code", valueCode = "116676008"),
      list(url = "value"),
      list(
        url = "subproperty",
        extension = list(
          list(url = "code", valueCode = "363698007"),
          list(url = "value")
        )
      )
    )
  )))

  # Act / Assert - the concept is left with no features rather than erroring.
  expect_equal(properties_to_list(coding), list())
})

test_that("a subproperty with no code is skipped", {
  # Arrange
  coding <- list(property = list(list(
    code = "609096000",
    extension = list(
      list(url = "subproperty", extension = list(
        list(url = "value", valueCode = "108369006")
      )),
      list(url = "subproperty", extension = list(
        list(url = "code", valueCode = "116676008"),
        list(url = "value", valueCode = "108369006")
      ))
    )
  )))

  # Act / Assert
  expect_equal(
    properties_to_list(coding), list("609096000.116676008" = "108369006")
  )
})

test_that("a composite property value is skipped", {
  # Arrange - a `valueCoding` has no scalar representation, so it cannot name
  # a feature.
  coding <- list(property = list(
    list(
      code = "116676008",
      valueCoding = list(system = "http://snomed.info/sct", code = "108369006")
    ),
    list(code = "363698007", valueCode = "39937001")
  ))

  # Act / Assert - the well-formed property beside it still contributes.
  expect_equal(
    properties_to_list(coding), list("363698007" = "39937001")
  )
})

test_that("a repeated property key keeps the last value, as a dictionary would", {
  # Arrange - two role groups carrying the same associated morphology key.
  coding <- list(property = list(
    list(code = "609096000", extension = list(list(
      url = "subproperty",
      extension = list(
        list(url = "code", valueCode = "116676008"),
        list(url = "value", valueCode = "108369006")
      )
    ))),
    list(code = "609096000", extension = list(list(
      url = "subproperty",
      extension = list(
        list(url = "code", valueCode = "116676008"),
        list(url = "value", valueCode = "1240414004")
      )
    )))
  ))

  # Act
  result <- properties_to_list(coding)

  # Assert - one entry, holding the later value: the same collapse Python's
  # dictionary performs.
  expect_equal(result, list("609096000.116676008" = "1240414004"))
})

# ---------------------------------------------------------------------------
# Vectorisation
# ---------------------------------------------------------------------------

test_that("character values become indicator features named key=value", {
  # Arrange - two concepts sharing a key but differing in value.
  lists <- list(
    list("116676008" = "108369006"),
    list("116676008" = "1240414004")
  )

  # Act
  result <- vectorize_properties(lists)

  # Assert - one column per distinct key/value pair, each cell a 1.
  expect_equal(
    result$feature_names,
    c("116676008=108369006", "116676008=1240414004")
  )
  expect_s4_class(result$matrix, "dgCMatrix")
  expect_equal(as.matrix(result$matrix), matrix(c(1, 0, 0, 1), nrow = 2, byrow = TRUE))
})

test_that("numeric values become one feature named by the key, carrying the number", {
  # Arrange
  lists <- list(list(severityScore = 2.5), list(severityScore = 4))

  # Act
  result <- vectorize_properties(lists)

  # Assert - no `=value` suffix, and the cell holds the value rather than a 1.
  expect_equal(result$feature_names, "severityScore")
  expect_equal(as.matrix(result$matrix), matrix(c(2.5, 4), ncol = 1))
})

test_that("a boolean value is carried as a number, as Python treats it", {
  # Arrange - a `valueBoolean` property. In Python a boolean is a number, so
  # DictVectorizer gives it a single unsuffixed feature.
  lists <- list(list(inactive = TRUE), list(inactive = FALSE))

  # Act
  result <- vectorize_properties(lists)

  # Assert
  expect_equal(result$feature_names, "inactive")
  expect_equal(as.matrix(result$matrix), matrix(c(1, 0), ncol = 1))
})

test_that("a concept with no properties gets a row of zeros", {
  # Arrange - the middle concept carries nothing.
  lists <- list(list(a = "x"), list(), list(a = "y", n = 2.5))

  # Act
  result <- vectorize_properties(lists)

  # Assert
  expect_equal(result$feature_names, c("a=x", "a=y", "n"))
  expect_equal(
    as.matrix(result$matrix),
    matrix(
      c(
        1, 0, 0,
        0, 0, 0,
        0, 1, 2.5
      ),
      nrow = 3, byrow = TRUE
    )
  )
})

test_that("no properties at all yields a matrix with one row per concept and no columns", {
  # Arrange / Act
  result <- vectorize_properties(list(list(), list()))

  # Assert - the widening is a no-op, but the row count still lines up with
  # the codes so that cbind() works.
  expect_equal(result$feature_names, character())
  expect_equal(dim(result$matrix), c(2L, 0L))
})

test_that("feature names are sorted in byte order rather than the user's locale", {
  # Arrange - under the en_AU (and en_US) collation R sorts these as
  # "_x=1", "a=1", "Z=1"; scikit-learn's DictVectorizer sorts by code point,
  # giving "Z=1", "_x=1", "a=1". A locale-collated sort therefore fails here.
  lists <- list(list(Z = "1", "_x" = "1", a = "1"))

  # Act
  result <- vectorize_properties(lists)

  # Assert - C locale order, and the columns follow the names.
  expect_equal(result$feature_names, c("Z=1", "_x=1", "a=1"))
  expect_equal(as.matrix(result$matrix), matrix(1, nrow = 1, ncol = 3))
})

test_that("the fixture property set vectorises to the documented features", {
  # Arrange - the five concepts of the property fixtures.
  lists <- fixture_properties("expand-properties-preadopt")

  # Act
  result <- vectorize_properties(lists)

  # Assert - the six distinct features documented in fixtures/README.md, in
  # byte order, with the numeric feature last and unsuffixed.
  expect_equal(
    result$feature_names,
    c(
      "116676008=108369006",
      "609096000.116676008.363698007=39937001",
      "609096000.116676008=108369006",
      "609096000.116676008=1240414004",
      "609096000.363698007=39937001",
      "severityScore"
    )
  )
  expect_equal(
    as.matrix(result$matrix),
    matrix(
      c(
        # 363346000
        0, 0, 0, 1, 1, 0,
        # 399981008
        0, 1, 1, 0, 0, 0,
        # 55342001
        1, 0, 0, 0, 0, 0,
        # 64572001
        0, 0, 0, 0, 0, 2.5,
        # 404684003
        0, 0, 0, 0, 0, 0
      ),
      nrow = 5, byrow = TRUE
    )
  )
})

# ---------------------------------------------------------------------------
# Widening the encoder
# ---------------------------------------------------------------------------

# The README example: the captured scope, encoded with the "Associated
# morphology" property.
build_property_encoder <- function() {
  suppressMessages(with_fixture_api(
    fhir_tx_encoder(
      scope = test_scope, tx_url = test_tx_url, properties = "116676008"
    ),
    expand = "expand-single-page",
    closure_update = "closure-update"
  ))
}

expected_property_features <- c(
  "609096000.116676008=108369006",
  "609096000.116676008=1240414004",
  "609096000.116676008=400177003"
)

test_that("requesting properties widens the encoding and appends the feature names", {
  # Arrange / Act
  encoder <- build_property_encoder()

  # Assert - the documented 6 x 9 encoding: six code columns then three
  # property columns.
  expect_equal(dim(encoder$encoded), c(6L, 9L))
  expect_s4_class(encoder$encoded, "dgCMatrix")
  expect_equal(
    encoder$feature_names, c(expected_codes, expected_property_features)
  )
  # The codes and displays are untouched by the widening.
  expect_equal(encoder$codes, expected_codes)
})

test_that("concepts without properties carry zeros in every property column", {
  # Arrange / Act - 404684003, 64572001 and 138875005 have no properties.
  encoder <- build_property_encoder()
  property_columns <- as.matrix(encoder$encoded[, 7:9, drop = FALSE])

  # Assert
  expect_equal(unname(property_columns[1, ]), c(0, 0, 0))
  expect_equal(unname(property_columns[2, ]), c(0, 0, 0))
  expect_equal(unname(property_columns[6, ]), c(0, 0, 0))
  # And the three concepts that do have one carry a single 1 each.
  expect_equal(unname(property_columns[3, ]), c(0, 1, 0))
  expect_equal(unname(property_columns[4, ]), c(0, 0, 1))
  expect_equal(unname(property_columns[5, ]), c(1, 0, 0))
})

test_that("the README example transforms to the documented 2 x 9 result", {
  # Arrange - "Neoplasm and/or hamartoma" then "Malignant neoplastic disease".
  encoder <- build_property_encoder()

  # Act
  result <- transform(encoder, matrix(c("399981008", "363346000"), ncol = 1))

  # Assert - exactly the rows printed in the repository README.
  expect_equal(dim(result), c(2L, 9L))
  expect_equal(
    as.matrix(result),
    matrix(
      c(
        1, 1, 0, 1, 0, 1, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 0, 1, 0
      ),
      nrow = 2, byrow = TRUE
    )
  )
})

test_that("property encoding progress is reported through a suppressible message", {
  # Arrange / Act
  emitted <- paste(capture_messages(with_fixture_api(
    fhir_tx_encoder(
      scope = test_scope, tx_url = test_tx_url, properties = "116676008"
    ),
    expand = "expand-single-page",
    closure_update = "closure-update"
  )), collapse = "")

  # Assert - the same landmark the Python implementation prints, with the
  # shape of the widened matrix.
  expect_match(emitted, "Encoding properties... (6, 9)", fixed = TRUE)
})

test_that("no property columns are added when no properties are requested", {
  # Arrange / Act - the same fixture carries properties, but they were not
  # asked for, so they must not appear.
  encoder <- suppressMessages(with_fixture_api(
    fhir_tx_encoder(scope = test_scope, tx_url = test_tx_url),
    expand = "expand-single-page",
    closure_update = "closure-update"
  ))
  emitted <- paste(capture_messages(with_fixture_api(
    fhir_tx_encoder(scope = test_scope, tx_url = test_tx_url),
    expand = "expand-single-page",
    closure_update = "closure-update"
  )), collapse = "")

  # Assert
  expect_equal(dim(encoder$encoded), c(6L, 6L))
  expect_equal(encoder$feature_names, expected_codes)
  expect_no_match(emitted, "Encoding properties", fixed = TRUE)
})
