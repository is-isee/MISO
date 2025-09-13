#pragma once

#include "grid_cpu.hpp"
#include "mpi_manager.hpp"

#if defined(__CUDACC__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

#if defined(__CUDACC__)
#define DEVICE __device__
#else
#define DEVICE
#endif

#if defined(__CUDACC__)
#define GLOBAL __global__
#else
#define GLOBAL
#endif

namespace bnd {
enum class Direction { X, Y, Z };
inline std::string direction_to_string(Direction direction) {
  switch (direction) {
  case Direction::X:
    return "x";
  case Direction::Y:
    return "y";
  case Direction::Z:
    return "z";
  default:
    return "Unknown";
  }
}
inline Direction string_to_direction(const std::string &str) {
  if (str == "x")
    return Direction::X;
  if (str == "y")
    return Direction::Y;
  if (str == "z")
    return Direction::Z;
  throw std::invalid_argument("Invalid direction string");
}

enum class Side { INNER, OUTER };
inline std::string side_to_string(Side side) {
  switch (side) {
  case Side::INNER:
    return "inner";
  case Side::OUTER:
    return "outer";
  default:
    return "Unknown";
  }
}
inline Side string_to_side(const std::string &str) {
  if (str == "inner")
    return Side::INNER;
  if (str == "outer")
    return Side::OUTER;
  throw std::invalid_argument("Invalid side string");
}

template <typename Real, typename GridType>
HOST_DEVICE inline void range_set(int &i0_, int &i1_, int &j0_, int &j1_,
                                  int &k0_, int &k1_, Direction direction,
                                  const GridType &grid) {
  i0_ = 0;
  i1_ = grid.i_total;
  j0_ = 0;
  j1_ = grid.j_total;
  k0_ = 0;
  k1_ = grid.k_total;
  switch (direction) {
  case Direction::X:
    i1_ = grid.i_margin;
    break;
  case Direction::Y:
    j1_ = grid.j_margin;
    break;
  case Direction::Z:
    k1_ = grid.k_margin;
    break;
  }
}

template <typename Real>
HOST_DEVICE inline void symmetric_index(int i, int i_total, int i_margin,
                                        int &i_ghst, int &i_trgt, Side side) {
  switch (side) {
  case Side::INNER:
    i_ghst = i;
    i_trgt = 2 * i_margin - i - 1;
    break;
  case Side::OUTER:
    i_ghst = i_total - i_margin + i;
    i_trgt = i_total - i_margin - i - 1;
    break;
  }
}

template <typename Real>
HOST_DEVICE inline void periodic_index(int i, int i_total, int i_margin,
                                       int &i_ghst, int &i_trgt, Side side) {
  switch (side) {
  case Side::INNER:
    i_ghst = i;
    i_trgt = i_total - 2 * i_margin + i;
    break;
  case Side::OUTER:
    i_ghst = i_total - i_margin + i;
    i_trgt = i_margin + i;
    break;
  }
}

template <typename Real>
bool is_physical_boundary(const Direction direction, const Side side,
                          const MPIManager<Real> &mpi) {
  switch (direction) {
  case Direction::X:
    return (side == Side::INNER) ? (mpi.x_procs_neg == MPI_PROC_NULL)
                                 : (mpi.x_procs_pos == MPI_PROC_NULL);
  case Direction::Y:
    return (side == Side::INNER) ? (mpi.y_procs_neg == MPI_PROC_NULL)
                                 : (mpi.y_procs_pos == MPI_PROC_NULL);
  case Direction::Z:
    return (side == Side::INNER) ? (mpi.z_procs_neg == MPI_PROC_NULL)
                                 : (mpi.z_procs_pos == MPI_PROC_NULL);
  }
  return false;
}
}  // namespace bnd
