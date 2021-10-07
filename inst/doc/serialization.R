## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = identical(Sys.getenv("TORCH_TEST", unset = "0"), "1")
)

## ----setup--------------------------------------------------------------------
library(torch)

## -----------------------------------------------------------------------------
x <- torch_randn(10, 10)
torch_save(x, "tensor.pt")
x_ <- torch_load("tensor.pt")

torch_allclose(x, x_)

## -----------------------------------------------------------------------------
module <- nn_module(
  "my_module",
  initialize = function() {
    self$fc1 <- nn_linear(10, 10)
    self$fc2 <- nn_linear(10, 1)
  },
  forward = function(x) {
    x %>% 
      self$fc1() %>% 
      self$fc2()
  }
)

model <- module()
torch_save(model, "model.pt")
model_ <- torch_load("model.pt")

# input tensor
x <- torch_randn(50, 10)
torch_allclose(model(x), model_(x))

## ----eval = FALSE-------------------------------------------------------------
#  state_dict <- load_state_dict(fpath)
#  model <- Model()
#  model$load_state_dict(state_dict)

## -----------------------------------------------------------------------------
model <- nn_linear(1, 1)
opt <- optim_adam(model$parameters)

train_x <- torch_randn(100, 1)
train_y <- torch_randn(100, 1)

loss <- nnf_mse_loss(model(train_x), train_y)
loss$backward()
opt$step()

# Now let's save the optimizer state
tmp <- tempfile()
torch_save(opt$state_dict(), tmp)

# And now let's create a new optimizer and load back
opt2 <- optim_adam(model$parameters)
opt2$load_state_dict(torch_load(tmp))

