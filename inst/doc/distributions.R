## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = identical(Sys.getenv("TORCH_TEST", unset = "0"), "1")
)

## ----setup--------------------------------------------------------------------
library(torch)
torch_manual_seed(1) # setting seed for reproducibility

## -----------------------------------------------------------------------------
n <- distr_normal(loc = 0, scale = 1)
n

## -----------------------------------------------------------------------------
n$sample()

## -----------------------------------------------------------------------------
n$sample(3)

## -----------------------------------------------------------------------------
n$log_prob(0)
log(dnorm(0)) # equivalent R code

## -----------------------------------------------------------------------------
n$log_prob(c(0, 2, 4))

## -----------------------------------------------------------------------------
b <- distr_bernoulli(probs = torch_tensor(c(0.25, 0.5, 0.75)))
b

## -----------------------------------------------------------------------------
b$sample()

## -----------------------------------------------------------------------------
b$sample(6)

## -----------------------------------------------------------------------------
x <- torch_randn(100, 1)
y <- 2*x + 1 + torch_randn(100, 1)

## -----------------------------------------------------------------------------
plot(as.numeric(x), as.numeric(y))

## -----------------------------------------------------------------------------
GaussianLinear <- nn_module(
  initialize = function() {
    # this linear predictor will estimate the mean of the normal distribution
    self$linear <- nn_linear(1, 1)
    # this parameter will hold the estimate of the variability
    self$scale <- nn_parameter(torch_ones(1))
  },
  forward = function(x) {
    # we estimate the mean
    loc <- self$linear(x)
    # return a normal distribution
    distr_normal(loc, self$scale)
  }
)

model <- GaussianLinear()

## -----------------------------------------------------------------------------
opt <- optim_sgd(model$parameters, lr = 0.1)

for (i in 1:100) {
  opt$zero_grad()
  d <- model(x)
  loss <- torch_mean(-d$log_prob(y))
  loss$backward()
  opt$step()
  if (i %% 10 == 0)
    cat("iter: ", i, " loss: ", loss$item(), "\n")
}

## -----------------------------------------------------------------------------
model$parameters

## -----------------------------------------------------------------------------
summary(glm(as.numeric(y) ~ as.numeric(x)))

