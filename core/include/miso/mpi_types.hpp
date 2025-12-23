#pragma once
#include <mpi.h>

template <typename Real> MPI_Datatype mpi_type();

template <> inline MPI_Datatype mpi_type<float>() { return MPI_FLOAT; }

template <> inline MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }
