mdlp <- function(x, ...) {
  UseMethod("mdlp")
}

mdlp.formula <- function(formula, data) {
  if (is.null(data) || is.na(data)) {
    stop("Argument data can't be NULL or NA.")
  }
  data_to_discr <- model.frame(formula, data)
  x <- data_to_discr[, -1]; y <- data_to_discr[, 1]
  mdlp(x, y)
}

mdlp.default <- function(x, y) {
  if (is.null(x) || is.na(x)) {
    stop("Argument x can't be NULL or NA.")
  } else if (is.null(y) || is.na(y)) {
    stop("Argument y can't be NULL or NA.")
  }
  print("Dupa")
  mdlpCpp(x, y)
}
