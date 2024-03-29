% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gen-namespace-docs.R, R/gen-namespace.R
\name{torch_lu_unpack}
\alias{torch_lu_unpack}
\title{Lu_unpack}
\usage{
torch_lu_unpack(LU_data, LU_pivots, unpack_data = TRUE, unpack_pivots = TRUE)
}
\arguments{
\item{LU_data}{(Tensor) – the packed LU factorization data}

\item{LU_pivots}{(Tensor) – the packed LU factorization pivots}

\item{unpack_data}{(logical) – flag indicating if the data should be unpacked. If FALSE, then the returned L and U are NULL Default: TRUE}

\item{unpack_pivots}{(logical) – flag indicating if the pivots should be unpacked into a permutation matrix P. If FALSE, then the returned P is None. Default: TRUE}
}
\description{
Lu_unpack
}
\section{lu_unpack(LU_data, LU_pivots, unpack_data = TRUE, unpack_pivots=TRUE) -> Tensor }{


Unpacks the data and pivots from a LU factorization of a tensor into tensors \code{L} and \code{U} and
a permutation tensor \code{P} such that \code{LU_data_and_pivots <- torch_lu(P$matmul(L)$matmul(U))}.
Returns a list of tensors as \verb{list(the P tensor (permutation matrix), the L tensor, the U tensor)}
}

