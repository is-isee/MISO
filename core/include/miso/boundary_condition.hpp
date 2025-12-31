#pragma once

#include <stdexcept>

#include <miso/array3d.hpp>
#include <miso/grid.hpp>
#include <miso/mpi_util.hpp>
#include <miso/types.hpp>
#include <miso/utility.hpp>
#ifdef USE_CUDA
#include <miso/cuda_util.cuh>
#endif  // USE_CUDA

namespace miso {
namespace bnd {

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

enum class Side { INNER, OUTER };

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

template <typename Real, typename GridType>
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

template <typename Real>
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

template <typename Real>
__host__ __device__ inline void periodic_index(int i, int i_total, int i_margin,
                                               int &i_ghst, int &i_trgt,
                                               Side side) {
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

template <typename Real>
void symmetric(Array3D<Real, backend::Host> &arr, const Grid<Real, backend::Host> &grid,
               Real sign, Direction direction, Side side) {
  int i0_, i1_, j0_, j1_, k0_, k1_;
  range_set<Real>(i0_, i1_, j0_, j1_, k0_, k1_, direction, grid);
  for (int i = i0_; i < i1_; ++i) {
    for (int j = j0_; j < j1_; ++j) {
      for (int k = k0_; k < k1_; ++k) {
        int i_ghst = i, i_trgt = i;
        int j_ghst = j, j_trgt = j;
        int k_ghst = k, k_trgt = k;
        switch (direction) {
        case Direction::X:
          symmetric_index<Real>(i, grid.i_total, grid.i_margin, i_ghst, i_trgt,
                                side);
          break;
        case Direction::Y:
          symmetric_index<Real>(j, grid.j_total, grid.j_margin, j_ghst, j_trgt,
                                side);
          break;
        case Direction::Z:
          symmetric_index<Real>(k, grid.k_total, grid.k_margin, k_ghst, k_trgt,
                                side);
          break;
        }
        arr(i_ghst, j_ghst, k_ghst) = sign * arr(i_trgt, j_trgt, k_trgt);
      }
    }
  }
};

#ifdef USE_CUDA
template <typename Real>
__global__ void symmetric_kernel(Array3DView<Real> arr, GridView<Real> grid,
                                 Real sign, Direction direction, Side side) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int i0_, i1_, j0_, j1_, k0_, k1_;
  range_set<Real>(i0_, i1_, j0_, j1_, k0_, k1_, direction, grid);

  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    int i_ghst = i, i_trgt = i;
    int j_ghst = j, j_trgt = j;
    int k_ghst = k, k_trgt = k;

    switch (direction) {
    case Direction::X:
      symmetric_index<Real>(i, grid.i_total, grid.i_margin, i_ghst, i_trgt, side);
      break;
    case Direction::Y:
      symmetric_index<Real>(j, grid.j_total, grid.j_margin, j_ghst, j_trgt, side);
      break;
    case Direction::Z:
      symmetric_index<Real>(k, grid.k_total, grid.k_margin, k_ghst, k_trgt, side);
      break;
    }
    arr(i_ghst, j_ghst, k_ghst) = sign * arr(i_trgt, j_trgt, k_trgt);
  }
}

template <typename Real>
void symmetric(Array3D<Real, backend::CUDA> &arr, const Grid<Real, backend::CUDA> &grid,
               Real sign, Direction direction, Side side) {
  dim3 block_dim(8, 8, 8);
  dim3 grid_dim((grid.i_total + block_dim.x - 1) / block_dim.x,
                (grid.j_total + block_dim.y - 1) / block_dim.y,
                (grid.k_total + block_dim.z - 1) / block_dim.z);

  symmetric_kernel<Real>
      <<<grid_dim, block_dim>>>(arr.view(), grid.view(), sign, direction, side);
  MISO_CUDA_CHECK(cudaGetLastError());
  MISO_CUDA_CHECK(cudaDeviceSynchronize());
};
#endif  // USE_CUDA

}  // namespace bnd
}  // namespace miso
