## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = identical(Sys.getenv("TORCH_TEST", unset = "0"), "1")
)

## ----setup--------------------------------------------------------------------
library(torch)

## -----------------------------------------------------------------------------
fn <- function(x) {
  torch_relu(x)
}

traced_fn <- jit_trace(fn, torch_tensor(c(-1, 0, 1)))

## -----------------------------------------------------------------------------
traced_fn$graph

## -----------------------------------------------------------------------------
traced_fn(torch_randn(3))

## -----------------------------------------------------------------------------
module <- nn_module(
  initialize = function() {
    self$linear1 <- nn_linear(10, 10)
    self$linear2 <- nn_linear(10, 1)
  },
  forward = function(x) {
    x %>% 
      self$linear1() %>% 
      nnf_relu() %>% 
      self$linear2()
  }
)
traced_module <- jit_trace(module(), torch_randn(10, 10))

## -----------------------------------------------------------------------------
traced_module(torch_randn(3, 10))

## ---- error=TRUE--------------------------------------------------------------
# fn does does an operation for each dimension of a tensor
fn <- function(x) {
  x %>% 
    torch_unbind(dim = 1) %>% 
    lapply(function(x) x$sum()) %>% 
    torch_stack(dim = 1)
}
# we trace using as an example a tensor with size (10, 5, 5)
traced_fn <- jit_trace(fn, torch_randn(10, 5, 5))
# applying it with a tensor with different size returns an error.
traced_fn(torch_randn(11, 5, 5))

## -----------------------------------------------------------------------------
traced_dropout <- jit_trace(nn_dropout(), torch_ones(5,5))
traced_dropout(torch_ones(3,3))
traced_dropout$eval()
# even after setting to eval mode, dropout is applied
traced_dropout(torch_ones(3,3))

## ---- error = TRUE------------------------------------------------------------
fn <- function(x, y) {
  x + y
}
jit_trace(fn, torch_tensor(1), 1)

## -----------------------------------------------------------------------------
tr <- jit_compile("
def fn (x: Tensor):
  return torch.relu(x)

")
tr$fn(torch_tensor(c(-1, 0, 1)))

## -----------------------------------------------------------------------------
fn <- function(x) {
  torch_relu(x)
}
tr_fn <- jit_trace(fn, torch_tensor(1))
jit_save(tr_fn, "path.pt")
loaded <- jit_load("path.pt")

## -----------------------------------------------------------------------------
loaded(torch_tensor(c(-1, 0, 1)))

