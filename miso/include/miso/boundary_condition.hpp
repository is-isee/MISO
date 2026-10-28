#pragma once

#include <stdexcept>

#include "array3d.hpp"
#include "backend.hpp"
#include "execution.hpp"
#include "grid.hpp"
#include "mpi_util.hpp"
#include "types.hpp"
#include "utility.hpp"

namespace miso {
namespace boundary_condition {

inline std::string direction_to_string(const Direction direction) {
  if (direction == Direction::X)
    return "x";
  if (direction == Direction::Y)
    return "y";
  if (direction == Direction::Z)
    return "z";
  throw std::logic_error("Not implemented.");
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

inline std::string side_to_string(Side side) {
  if (side == Side::INNER)
    return "inner";
  if (side == Side::OUTER)
    return "outer";
  throw std::logic_error("Not implemented.");
}

inline Side string_to_side(const std::string &str) {
  if (str == "inner")
    return Side::INNER;
  if (str == "outer")
    return Side::OUTER;
  throw std::invalid_argument("Invalid side string");
}

template <typename GridType>
__host__ __device__ inline void range_set(int &i0_, int &i1_, int &j0_, int &j1_,
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

__host__ __device__ inline void symmetric_index(int i, int i_total, int i_margin,
                                                int &i_ghst, int &i_trgt,
                                                Side side) {
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

inline bool is_physical_boundary(const Direction direction, const Side side,
                                 const mpi::Shape &mpi_shape) {
  switch (direction) {
  case Direction::X:
    return (side == Side::INNER) ? (mpi_shape.x_procs_neg == MPI_PROC_NULL)
                                 : (mpi_shape.x_procs_pos == MPI_PROC_NULL);
  case Direction::Y:
    return (side == Side::INNER) ? (mpi_shape.y_procs_neg == MPI_PROC_NULL)
                                 : (mpi_shape.y_procs_pos == MPI_PROC_NULL);
  case Direction::Z:
    return (side == Side::INNER) ? (mpi_shape.z_procs_neg == MPI_PROC_NULL)
                                 : (mpi_shape.z_procs_pos == MPI_PROC_NULL);
  }
  return false;
}

/// @brief Apply symmetric boundary condition to an given 3D array.
/// @details One implementation for all backends. The loop runs over the ghost
/// cells of the given side only.
template <typename Real, typename Backend>
void symmetric(Backend, Array3DView<Real> arr, GridView<const Real> grid,
               Sign sign, Direction direction, Side side) {
  int i0_, i1_, j0_, j1_, k0_, k1_;
  range_set(i0_, i1_, j0_, j1_, k0_, k1_, direction, grid);
  const Range3D range{{i0_, i1_}, {j0_, j1_}, {k0_, k1_}};
  const Real sign_f = (sign == Sign::Pos) ? Real(1.0) : Real(-1.0);
  for_each(
      Backend{}, range, MISO_LAMBDA(int i, int j, int k) {
        int i_ghst = i, i_trgt = i;
        int j_ghst = j, j_trgt = j;
        int k_ghst = k, k_trgt = k;
        switch (direction) {
        case Direction::X:
          symmetric_index(i, grid.i_total, grid.i_margin, i_ghst, i_trgt, side);
          break;
        case Direction::Y:
          symmetric_index(j, grid.j_total, grid.j_margin, j_ghst, j_trgt, side);
          break;
        case Direction::Z:
          symmetric_index(k, grid.k_total, grid.k_margin, k_ghst, k_trgt, side);
          break;
        }
        arr(i_ghst, j_ghst, k_ghst) = sign_f * arr(i_trgt, j_trgt, k_trgt);
      });
}

}  // namespace boundary_condition
}  // namespace miso
