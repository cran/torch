#include "torch_types.h"
#include "utils.h"

void call_r_gc ()
{
  static Rcpp::Function r_gc("gc");
  r_gc(Rcpp::Named("full")=false);
}

// [[Rcpp::export]]
void cpp_set_lantern_allocator (uint64_t threshold_call_gc = 4000)
{
  set_lantern_allocator(&call_r_gc, threshold_call_gc);
}
