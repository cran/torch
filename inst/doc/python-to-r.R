## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = identical(Sys.getenv("TORCH_TEST", unset = "0"), "1")
)

## ----eval=FALSE---------------------------------------------------------------
#  library(torch)
#  
#  #Make up some test data
#  #note that proper installation of torch will yield no errors when we run
#  #this code
#  y <- torch_tensor(array(runif(8),dim = c(2,2,2)),dtype = torch_float64())
#  
#  #Note the identical names between the Python class definition and our
#  #class definition
#  simpleMLP <- torch::nn_module(
#    "simpleMLP",
#    initialize = function(){
#  
#      self$modelFit <- nn_sequential(nn_linear(100,20),
#                                     nn_relu(),
#                                     nn_linear(20,1),
#                                     nn_sigmoid())
#  
#    },
#    forward = function(x){
#      self$modelFit(x)
#      }
#  )
#  
#  
#  model <- simpleMLP()
#  
#  state_dict <- torch::load_state_dict("p/babyTest.pth")
#  model$load_state_dict(state_dict)
#  
#  #Note that the dtype set in R has to match the made up data from Python
#  #More generally if reading new data into R you must ensure that it matches the
#  #dtype that the model was trained with in Python
#  newData = torch_tensor(array(rnorm(n=1000),dim=c(10,100)),dtype=torch_float32())
#  
#  predictMe <- model(newData)

