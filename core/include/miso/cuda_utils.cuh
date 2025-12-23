#pragma once

#include <miso/cuda_compat.hpp>
#include <miso/grid.hpp>

namespace miso {

template <typename Real> struct CudaKernelShape {
  dim3 block_dim;
  dim3 grid_dim;
  dim3 grid_dim_x_margin, grid_dim_y_margin, grid_dim_z_margin;

  explicit CudaKernelShape(const Grid<Real> &grid) {
    const auto div_up = [](int n, int d) { return (n + d - 1) / d; };

    block_dim = dim3(8, 8, 8);
    // clang-format off
    grid_dim = dim3(div_up(grid.i_total, block_dim.x),
                    div_up(grid.j_total, block_dim.y),
                    div_up(grid.k_total, block_dim.z));
    grid_dim_x_margin = dim3(div_up(grid.i_margin, block_dim.x),
                             div_up(grid.j_total, block_dim.y),
                             div_up(grid.k_total, block_dim.z));
    grid_dim_y_margin = dim3(div_up(grid.i_total, block_dim.x),
                             div_up(grid.j_margin, block_dim.y),
                             div_up(grid.k_total, block_dim.z));
    grid_dim_z_margin = dim3(div_up(grid.i_total, block_dim.x),
                             div_up(grid.j_total, block_dim.y),
                             div_up(grid.k_margin, block_dim.z));
    // clang-format on
  }
};

}  // namespace miso
