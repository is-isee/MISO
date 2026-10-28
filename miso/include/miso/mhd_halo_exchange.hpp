#pragma once

#include "array3d.hpp"
#include "array4d.hpp"
#include "backend.hpp"
#include "cuda_compat.hpp"
#include "execution.hpp"
#include "grid.hpp"
#include "mhd_core.hpp"
#include "mhd_fields.hpp"
#include "mpi_util.hpp"
#include "types.hpp"

namespace miso {
namespace mhd {

/// @brief The m-th variable of the MHD fields (ro, vx, vy, vz, bx, by, bz, ei,
/// ph), the order of the halo exchange buffers.
template <typename T>
__host__ __device__ inline Array3DView<T> field_at(const FieldsView<T> &qq,
                                                   int m) {
  switch (m) {
  case 0:
    return qq.ro;
  case 1:
    return qq.vx;
  case 2:
    return qq.vy;
  case 3:
    return qq.vz;
  case 4:
    return qq.bx;
  case 5:
    return qq.by;
  case 6:
    return qq.bz;
  case 7:
    return qq.ei;
  default:
    return qq.ph;
  }
}

/// @brief Halo exchange of the MHD fields by MPI.
/// @details One implementation for all backends. The buffers are arrays of
/// the backend; with the CUDA backend, device pointers are passed to MPI
/// (CUDA-aware MPI).
template <typename Real, typename Backend> struct HaloExchanger {
  // MPI communication buffers (i, j, k, variable)
  Array4D<Real, Backend> recv_x_pos, recv_x_neg;
  Array4D<Real, Backend> recv_y_pos, recv_y_neg;
  Array4D<Real, Backend> recv_z_pos, recv_z_neg;
  Array4D<Real, Backend> send_x_pos, send_x_neg;
  Array4D<Real, Backend> send_y_pos, send_y_neg;
  Array4D<Real, Backend> send_z_pos, send_z_neg;

  Grid<Real, Backend> &grid;
  mpi::Shape &mpi_shape;

  HaloExchanger(Grid<Real, Backend> &grid, ExecContext<Real, Backend> &exec_ctx)
      : recv_x_pos(grid.i_margin, grid.j_total, grid.k_total, n_fields),
        recv_x_neg(grid.i_margin, grid.j_total, grid.k_total, n_fields),
        recv_y_pos(grid.i_total, grid.j_margin, grid.k_total, n_fields),
        recv_y_neg(grid.i_total, grid.j_margin, grid.k_total, n_fields),
        recv_z_pos(grid.i_total, grid.j_total, grid.k_margin, n_fields),
        recv_z_neg(grid.i_total, grid.j_total, grid.k_margin, n_fields),
        send_x_pos(grid.i_margin, grid.j_total, grid.k_total, n_fields),
        send_x_neg(grid.i_margin, grid.j_total, grid.k_total, n_fields),
        send_y_pos(grid.i_total, grid.j_margin, grid.k_total, n_fields),
        send_y_neg(grid.i_total, grid.j_margin, grid.k_total, n_fields),
        send_z_pos(grid.i_total, grid.j_total, grid.k_margin, n_fields),
        send_z_neg(grid.i_total, grid.j_total, grid.k_margin, n_fields),
        grid(grid), mpi_shape(exec_ctx.mpi_shape) {}

  /// @brief Copy the cells in `range` (offset by `src` in the direction) into
  /// the buffer, or the buffer back into them (`to_buffer = false`).
  /// @details The buffer index is the cell index relative to the beginning of
  /// `range`.
  static void copy(FieldsView<Real> qq, Array4DView<Real> buf, Range3D range,
                   bool to_buffer) {
    const int i0 = range.range0.begin, j0 = range.range1.begin,
              k0 = range.range2.begin;
    if (to_buffer) {
      for_each(
          Backend{}, range, MISO_LAMBDA(int i, int j, int k) {
            for (int m = 0; m < n_fields; ++m)
              buf(i - i0, j - j0, k - k0, m) = field_at(qq, m)(i, j, k);
          });
    } else {
      for_each(
          Backend{}, range, MISO_LAMBDA(int i, int j, int k) {
            for (int m = 0; m < n_fields; ++m)
              field_at(qq, m)(i, j, k) = buf(i - i0, j - j0, k - k0, m);
          });
    }
  }

  /// @brief Exchange the ghost cells in one direction with both neighbors.
  /// @param send_pos, send_neg Ranges of the cells sent to the neighbors
  /// @param recv_pos, recv_neg Ranges of the ghost cells received from them
  void exchange(FieldsView<Real> qq, int rank_pos, int rank_neg, int tag,
                Array4D<Real, Backend> &buf_send_pos,
                Array4D<Real, Backend> &buf_send_neg,
                Array4D<Real, Backend> &buf_recv_pos,
                Array4D<Real, Backend> &buf_recv_neg, Range3D send_pos,
                Range3D send_neg, Range3D recv_pos, Range3D recv_neg) {
    const bool has_pos = rank_pos != MPI_PROC_NULL;
    const bool has_neg = rank_neg != MPI_PROC_NULL;
    if (!has_pos && !has_neg)
      return;

    if (has_pos)
      copy(qq, buf_send_pos.view(), send_pos, true);
    if (has_neg)
      copy(qq, buf_send_neg.view(), send_neg, true);
    // The send buffers must be filled before MPI reads them. This also
    // ensures that no kernel still reads the receive buffers.
    fence(Backend{});

    MPI_Request reqs[4];
    int req_count = 0;
    const auto type = mpi::data_type<Real>();
    if (has_pos) {
      MPI_Isend(buf_send_pos.data(), buf_send_pos.size(), type, rank_pos, tag,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(buf_recv_pos.data(), buf_recv_pos.size(), type, rank_pos,
                tag + 100, mpi_shape.cart_comm, &reqs[req_count++]);
    }
    if (has_neg) {
      MPI_Isend(buf_send_neg.data(), buf_send_neg.size(), type, rank_neg,
                tag + 100, mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(buf_recv_neg.data(), buf_recv_neg.size(), type, rank_neg, tag,
                mpi_shape.cart_comm, &reqs[req_count++]);
    }
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

    if (has_pos)
      copy(qq, buf_recv_pos.view(), recv_pos, false);
    if (has_neg)
      copy(qq, buf_recv_neg.view(), recv_neg, false);
  }

  void apply(Fields<Real, Backend> &qq_trgt) {
    const auto qq = qq_trgt.view();
    const int it = grid.i_total, jt = grid.j_total, kt = grid.k_total;
    const int im = grid.i_margin, jm = grid.j_margin, km = grid.k_margin;

    // Exchange one direction at a time so that edge and corner ghost cells
    // are filled with up-to-date values (the y/z buffers include the
    // ghost cells received in the previous directions).
    if (grid.i_size > 1) {
      exchange(qq, mpi_shape.x_procs_pos, mpi_shape.x_procs_neg, 100, send_x_pos,
               send_x_neg, recv_x_pos, recv_x_neg,
               {{it - 2 * im, it - im}, {0, jt}, {0, kt}},
               {{im, 2 * im}, {0, jt}, {0, kt}},  //
               {{it - im, it}, {0, jt}, {0, kt}}, {{0, im}, {0, jt}, {0, kt}});
    }
    if (grid.j_size > 1) {
      exchange(qq, mpi_shape.y_procs_pos, mpi_shape.y_procs_neg, 300, send_y_pos,
               send_y_neg, recv_y_pos, recv_y_neg,
               {{0, it}, {jt - 2 * jm, jt - jm}, {0, kt}},
               {{0, it}, {jm, 2 * jm}, {0, kt}},  //
               {{0, it}, {jt - jm, jt}, {0, kt}}, {{0, it}, {0, jm}, {0, kt}});
    }
    if (grid.k_size > 1) {
      exchange(qq, mpi_shape.z_procs_pos, mpi_shape.z_procs_neg, 500, send_z_pos,
               send_z_neg, recv_z_pos, recv_z_neg,
               {{0, it}, {0, jt}, {kt - 2 * km, kt - km}},
               {{0, it}, {0, jt}, {km, 2 * km}},  //
               {{0, it}, {0, jt}, {kt - km, kt}}, {{0, it}, {0, jt}, {0, km}});
    }
  }
};

}  // namespace mhd
}  // namespace miso
