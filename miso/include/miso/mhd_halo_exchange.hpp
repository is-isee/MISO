#pragma once

#include <array>

#include <miso/array3d.hpp>
#include <miso/array4d.hpp>
#include <miso/backend.hpp>
#include <miso/grid.hpp>
#include <miso/mhd_buffers.hpp>
#include <miso/mhd_core.hpp>
#include <miso/mhd_fields.hpp>
#include <miso/mpi_util.hpp>
#ifdef __CUDACC__
#include <miso/cuda_util.cuh>
#endif  // __CUDACC__

namespace miso {
namespace mhd {

template <typename Real, typename Backend = backend::Host>
struct HaloExchanger {};

template <typename Real> struct HaloExchanger<Real, backend::Host> {
  // MPI communication buffers
  Array4D<Real> recv_x_pos, recv_x_neg;
  Array4D<Real> recv_y_pos, recv_y_neg;
  Array4D<Real> recv_z_pos, recv_z_neg;
  Array4D<Real> send_x_pos, send_x_neg;
  Array4D<Real> send_y_pos, send_y_neg;
  Array4D<Real> send_z_pos, send_z_neg;

  Grid<Real, backend::Host> &grid;
  mpi::Shape &mpi_shape;

  HaloExchanger(Grid<Real, backend::Host> &grid,
                ExecContext<backend::Host> &exec_ctx)
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

  void apply(Fields<Real, backend::Host> &qq_trgt) {
    std::array<Array3D<Real, backend::Host> *, 9> vars = {
        &qq_trgt.ro, &qq_trgt.vx, &qq_trgt.vy, &qq_trgt.vz, &qq_trgt.bx,
        &qq_trgt.by, &qq_trgt.bz, &qq_trgt.ei, &qq_trgt.ph};
    MPI_Request reqs[12];
    int req_count = 0;

    // ################
    // positive x-direction
    if (mpi_shape.x_procs_pos != MPI_PROC_NULL && grid.i_size > 1) {
      for (int i = 0; i < grid.i_margin; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          for (int k = 0; k < grid.k_total; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_x_pos(i, j, k, v) =
                  (*vars[v])(grid.i_total - 2 * grid.i_margin + i, j, k);
            }
          }
        }
      }
      MPI_Isend(send_x_pos.data(), send_x_pos.size(), mpi::data_type<Real>(),
                mpi_shape.x_procs_pos, 100, mpi_shape.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(recv_x_pos.data(), recv_x_pos.size(), mpi::data_type<Real>(),
                mpi_shape.x_procs_pos, 200, mpi_shape.cart_comm,
                &reqs[req_count++]);
    }

    // negative x-direction
    if (mpi_shape.x_procs_neg != MPI_PROC_NULL && grid.i_size > 1) {
      for (int i = 0; i < grid.i_margin; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          for (int k = 0; k < grid.k_total; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_x_neg(i, j, k, v) = (*vars[v])(grid.i_margin + i, j, k);
            }
          }
        }
      }
      MPI_Isend(send_x_neg.data(), send_x_neg.size(), mpi::data_type<Real>(),
                mpi_shape.x_procs_neg, 200, mpi_shape.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(recv_x_neg.data(), recv_x_neg.size(), mpi::data_type<Real>(),
                mpi_shape.x_procs_neg, 100, mpi_shape.cart_comm,
                &reqs[req_count++]);
    }

    // ################
    // positive y-direction
    if (mpi_shape.y_procs_pos != MPI_PROC_NULL && grid.j_size > 1) {
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_margin; ++j) {
          for (int k = 0; k < grid.k_total; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_y_pos(i, j, k, v) =
                  (*vars[v])(i, grid.j_total - 2 * grid.j_margin + j, k);
            }
          }
        }
      }
      MPI_Isend(send_y_pos.data(), send_y_pos.size(), mpi::data_type<Real>(),
                mpi_shape.y_procs_pos, 300, mpi_shape.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(recv_y_pos.data(), recv_y_pos.size(), mpi::data_type<Real>(),
                mpi_shape.y_procs_pos, 400, mpi_shape.cart_comm,
                &reqs[req_count++]);
    }

    // negative y-direction
    if (mpi_shape.y_procs_neg != MPI_PROC_NULL && grid.j_size > 1) {
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_margin; ++j) {
          for (int k = 0; k < grid.k_total; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_y_neg(i, j, k, v) = (*vars[v])(i, grid.j_margin + j, k);
            }
          }
        }
      }
      MPI_Isend(send_y_neg.data(), send_y_neg.size(), mpi::data_type<Real>(),
                mpi_shape.y_procs_neg, 400, mpi_shape.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(recv_y_neg.data(), recv_y_neg.size(), mpi::data_type<Real>(),
                mpi_shape.y_procs_neg, 300, mpi_shape.cart_comm,
                &reqs[req_count++]);
    }

    // ################
    // positive z-direction
    if (mpi_shape.z_procs_pos != MPI_PROC_NULL && grid.k_size > 1) {
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          for (int k = 0; k < grid.k_margin; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_z_pos(i, j, k, v) =
                  (*vars[v])(i, j, grid.k_total - 2 * grid.k_margin + k);
            }
          }
        }
      }
      MPI_Isend(send_z_pos.data(), send_z_pos.size(), mpi::data_type<Real>(),
                mpi_shape.z_procs_pos, 500, mpi_shape.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(recv_z_pos.data(), recv_z_pos.size(), mpi::data_type<Real>(),
                mpi_shape.z_procs_pos, 600, mpi_shape.cart_comm,
                &reqs[req_count++]);
    }

    // negative z-direction
    if (mpi_shape.z_procs_neg != MPI_PROC_NULL && grid.k_size > 1) {
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          for (int k = 0; k < grid.k_margin; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_z_neg(i, j, k, v) = (*vars[v])(i, j, grid.k_margin + k);
            }
          }
        }
      }
      MPI_Isend(send_z_neg.data(), send_z_neg.size(), mpi::data_type<Real>(),
                mpi_shape.z_procs_neg, 600, mpi_shape.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(recv_z_neg.data(), recv_z_neg.size(), mpi::data_type<Real>(),
                mpi_shape.z_procs_neg, 500, mpi_shape.cart_comm,
                &reqs[req_count++]);
    }

    if (req_count > 0) {
      MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    }

    // ################
    // positive x-direction
    if (mpi_shape.x_procs_pos != MPI_PROC_NULL && grid.i_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_margin; ++i) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              (*vars[v])(grid.i_total - grid.i_margin + i, j, k) =
                  recv_x_pos(i, j, k, v);
            }
          }
        }
      }
    }

    // negative x-direction
    if (mpi_shape.x_procs_neg != MPI_PROC_NULL && grid.i_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_margin; ++i) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              (*vars[v])(i, j, k) = recv_x_neg(i, j, k, v);
            }
          }
        }
      }
    }

    // ################
    // positive y-direction
    if (mpi_shape.y_procs_pos != MPI_PROC_NULL && grid.j_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_total; ++i) {
          for (int j = 0; j < grid.j_margin; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              (*vars[v])(i, grid.j_total - grid.j_margin + j, k) =
                  recv_y_pos(i, j, k, v);
            }
          }
        }
      }
    }

    // negative y-direction
    if (mpi_shape.y_procs_neg != MPI_PROC_NULL && grid.j_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_total; ++i) {
          for (int j = 0; j < grid.j_margin; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              (*vars[v])(i, j, k) = recv_y_neg(i, j, k, v);
            }
          }
        }
      }
    }

    // ################
    // positive z-direction
    if (mpi_shape.z_procs_pos != MPI_PROC_NULL && grid.k_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_total; ++i) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_margin; ++k) {
              (*vars[v])(i, j, grid.k_total - grid.k_margin + k) =
                  recv_z_pos(i, j, k, v);
            }
          }
        }
      }
    }

    // negative z-direction
    if (mpi_shape.z_procs_neg != MPI_PROC_NULL && grid.k_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_total; ++i) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_margin; ++k) {
              (*vars[v])(i, j, k) = recv_z_neg(i, j, k, v);
            }
          }
        }
      }
    }
  }
};

#ifdef __CUDACC__
// ##################
// x-direction send
template <typename Real>
__global__ void pack_x_send(BuffersView<Real> buff,
                            const FieldsView<Real> qq_trgt,
                            const GridView<Real> grid, Face face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= grid.i_margin || j >= grid.j_total || k >= grid.k_total)
    return;

  Real *__restrict__ send_buff =
      (face == Face::Pos) ? buff.send_x_pos : buff.send_x_neg;

  const int i_src = (face == Face::Pos) ? (grid.i_total - 2 * grid.i_margin + i)
                                        : (grid.i_margin + i);
  const size_t src_idx = grid.idx(i_src, j, k);
  const size_t buf_idx = ((i * grid.j_total + j) * grid.k_total + k) * n_fields;

  const Real *__restrict__ var[n_fields] = {
      qq_trgt.ro.data(), qq_trgt.vx.data(), qq_trgt.vy.data(),
      qq_trgt.vz.data(), qq_trgt.bx.data(), qq_trgt.by.data(),
      qq_trgt.bz.data(), qq_trgt.ei.data(), qq_trgt.ph.data()};

#pragma unroll
  for (int m = 0; m < n_fields; ++m) {
    send_buff[buf_idx + m] = var[m][src_idx];
  }
};

// y-direction send
template <typename Real>
__global__ void pack_y_send(BuffersView<Real> buff,
                            const FieldsView<Real> qq_trgt,
                            const GridView<Real> grid, Face face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= grid.i_total || j >= grid.j_margin || k >= grid.k_total)
    return;

  Real *__restrict__ send_buff =
      (face == Face::Pos) ? buff.send_y_pos : buff.send_y_neg;

  const int j_src = (face == Face::Pos) ? (grid.j_total - 2 * grid.j_margin + j)
                                        : (grid.j_margin + j);
  const size_t src_idx = grid.idx(i, j_src, k);
  const size_t buf_idx = ((i * grid.j_margin + j) * grid.k_total + k) * n_fields;

  const Real *__restrict__ var[n_fields] = {
      qq_trgt.ro.data(), qq_trgt.vx.data(), qq_trgt.vy.data(),
      qq_trgt.vz.data(), qq_trgt.bx.data(), qq_trgt.by.data(),
      qq_trgt.bz.data(), qq_trgt.ei.data(), qq_trgt.ph.data()};

#pragma unroll
  for (int m = 0; m < n_fields; ++m) {
    send_buff[buf_idx + m] = var[m][src_idx];
  }
};

// z-direction send
template <typename Real>
__global__ void pack_z_send(BuffersView<Real> buff,
                            const FieldsView<Real> qq_trgt,
                            const GridView<Real> grid, Face face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= grid.i_total || j >= grid.j_total || k >= grid.k_margin)
    return;

  Real *__restrict__ send_buff =
      (face == Face::Pos) ? buff.send_z_pos : buff.send_z_neg;

  const int k_src = (face == Face::Pos) ? (grid.k_total - 2 * grid.k_margin + k)
                                        : (grid.k_margin + k);
  const size_t src_idx = grid.idx(i, j, k_src);
  const size_t buf_idx = ((i * grid.j_total + j) * grid.k_margin + k) * n_fields;

  const Real *__restrict__ var[n_fields] = {
      qq_trgt.ro.data(), qq_trgt.vx.data(), qq_trgt.vy.data(),
      qq_trgt.vz.data(), qq_trgt.bx.data(), qq_trgt.by.data(),
      qq_trgt.bz.data(), qq_trgt.ei.data(), qq_trgt.ph.data()};

#pragma unroll
  for (int m = 0; m < n_fields; ++m) {
    send_buff[buf_idx + m] = var[m][src_idx];
  }
};

// ##################
// x-direction recv
template <typename Real>
__global__ void unpack_x_recv(FieldsView<Real> qq_trgt,
                              const BuffersView<Real> buff,
                              const GridView<Real> grid, Face face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= grid.i_margin || j >= grid.j_total || k >= grid.k_total)
    return;
  Real *__restrict__ recv_buff =
      (face == Face::Pos) ? buff.recv_x_pos : buff.recv_x_neg;
  const int i_tgt = (face == Face::Pos) ? grid.i_total - grid.i_margin + i : i;

  const size_t tgt_idx = grid.idx(i_tgt, j, k);
  const size_t buf_idx = ((i * grid.j_total + j) * grid.k_total + k) * n_fields;

  Real *__restrict__ var[n_fields] = {
      qq_trgt.ro.data(), qq_trgt.vx.data(), qq_trgt.vy.data(),
      qq_trgt.vz.data(), qq_trgt.bx.data(), qq_trgt.by.data(),
      qq_trgt.bz.data(), qq_trgt.ei.data(), qq_trgt.ph.data()};

#pragma unroll
  for (int m = 0; m < n_fields; ++m) {
    var[m][tgt_idx] = recv_buff[buf_idx + m];
  }
};

// y-direction recv
template <typename Real>
__global__ void unpack_y_recv(FieldsView<Real> qq_trgt,
                              const BuffersView<Real> buff,
                              const GridView<Real> grid, Face face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= grid.i_total || j >= grid.j_margin || k >= grid.k_total)
    return;
  Real *__restrict__ recv_buff =
      (face == Face::Pos) ? buff.recv_y_pos : buff.recv_y_neg;
  const int j_tgt = (face == Face::Pos) ? grid.j_total - grid.j_margin + j : j;

  const size_t tgt_idx = grid.idx(i, j_tgt, k);
  const size_t buf_idx = ((i * grid.j_margin + j) * grid.k_total + k) * n_fields;

  Real *__restrict__ var[n_fields] = {
      qq_trgt.ro.data(), qq_trgt.vx.data(), qq_trgt.vy.data(),
      qq_trgt.vz.data(), qq_trgt.bx.data(), qq_trgt.by.data(),
      qq_trgt.bz.data(), qq_trgt.ei.data(), qq_trgt.ph.data()};

#pragma unroll
  for (int m = 0; m < n_fields; ++m) {
    var[m][tgt_idx] = recv_buff[buf_idx + m];
  }
};

// z-direction recv
template <typename Real>
__global__ void unpack_z_recv(FieldsView<Real> qq_trgt,
                              const BuffersView<Real> buff,
                              const GridView<Real> grid, Face face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= grid.i_total || j >= grid.j_total || k >= grid.k_margin)
    return;
  Real *__restrict__ recv_buff =
      (face == Face::Pos) ? buff.recv_z_pos : buff.recv_z_neg;
  const int k_tgt = (face == Face::Pos) ? grid.k_total - grid.k_margin + k : k;

  const size_t tgt_idx = grid.idx(i, j, k_tgt);
  const size_t buf_idx = ((i * grid.j_total + j) * grid.k_margin + k) * n_fields;

  Real *__restrict__ var[n_fields] = {
      qq_trgt.ro.data(), qq_trgt.vx.data(), qq_trgt.vy.data(),
      qq_trgt.vz.data(), qq_trgt.bx.data(), qq_trgt.by.data(),
      qq_trgt.bz.data(), qq_trgt.ei.data(), qq_trgt.ph.data()};

#pragma unroll
  for (int m = 0; m < n_fields; ++m) {
    var[m][tgt_idx] = recv_buff[buf_idx + m];
  }
};

template <typename Real> struct HaloExchanger<Real, backend::CUDA> {
  Buffers<Real, backend::CUDA> buff;
  Grid<Real, backend::CUDA> &grid;
  mpi::Shape &mpi_shape;
  cuda::KernelShape3D &cu_shape;

  explicit HaloExchanger(Grid<Real, backend::CUDA> &grid,
                         ExecContext<backend::CUDA> &exec_ctx)
      : buff(grid), grid(grid), mpi_shape(exec_ctx.mpi_shape),
        cu_shape(exec_ctx.cu_shape) {}

  void apply(Fields<Real, backend::CUDA> &qq_trgt) {
    MPI_Request reqs[12];
    int req_count = 0;

    // ##################
    // positive x-direction
    if (mpi_shape.x_procs_pos != MPI_PROC_NULL && grid.i_total > 1) {
      pack_x_send<<<cu_shape.grid_dim_x_margin, cu_shape.block_dim>>>(
          buff.view(), qq_trgt.view(), grid.view(), Face::Pos);
      MISO_CUDA_CHECK(cudaGetLastError());
      MISO_CUDA_CHECK(cudaDeviceSynchronize());

      MPI_Isend(buff.send_x_pos,
                grid.i_margin * grid.j_total * grid.k_total * n_fields,
                mpi::data_type<Real>(), mpi_shape.x_procs_pos, 100,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(buff.recv_x_pos,
                grid.i_margin * grid.j_total * grid.k_total * n_fields,
                mpi::data_type<Real>(), mpi_shape.x_procs_pos, 200,
                mpi_shape.cart_comm, &reqs[req_count++]);
    }

    // positive y-direction
    if (mpi_shape.y_procs_pos != MPI_PROC_NULL && grid.j_total > 1) {
      pack_y_send<<<cu_shape.grid_dim_y_margin, cu_shape.block_dim>>>(
          buff.view(), qq_trgt.view(), grid.view(), Face::Pos);
      MISO_CUDA_CHECK(cudaGetLastError());
      MISO_CUDA_CHECK(cudaDeviceSynchronize());

      MPI_Isend(buff.send_y_pos,
                grid.i_total * grid.j_margin * grid.k_total * n_fields,
                mpi::data_type<Real>(), mpi_shape.y_procs_pos, 300,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(buff.recv_y_pos,
                grid.i_total * grid.j_margin * grid.k_total * n_fields,
                mpi::data_type<Real>(), mpi_shape.y_procs_pos, 400,
                mpi_shape.cart_comm, &reqs[req_count++]);
    }

    // positive z-direction
    if (mpi_shape.z_procs_pos != MPI_PROC_NULL && grid.k_total > 1) {
      pack_z_send<<<cu_shape.grid_dim_z_margin, cu_shape.block_dim>>>(
          buff.view(), qq_trgt.view(), grid.view(), Face::Pos);
      MISO_CUDA_CHECK(cudaGetLastError());
      MISO_CUDA_CHECK(cudaDeviceSynchronize());

      MPI_Isend(buff.send_z_pos,
                grid.i_total * grid.j_total * grid.k_margin * n_fields,
                mpi::data_type<Real>(), mpi_shape.z_procs_pos, 500,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(buff.recv_z_pos,
                grid.i_total * grid.j_total * grid.k_margin * n_fields,
                mpi::data_type<Real>(), mpi_shape.z_procs_pos, 600,
                mpi_shape.cart_comm, &reqs[req_count++]);
    }

    // ##################
    // negative x-direction
    if (mpi_shape.x_procs_neg != MPI_PROC_NULL && grid.i_total > 1) {
      pack_x_send<<<cu_shape.grid_dim_x_margin, cu_shape.block_dim>>>(
          buff.view(), qq_trgt.view(), grid.view(), Face::Neg);
      MISO_CUDA_CHECK(cudaGetLastError());
      MISO_CUDA_CHECK(cudaDeviceSynchronize());
      MPI_Isend(buff.send_x_neg,
                grid.i_margin * grid.j_total * grid.k_total * n_fields,
                mpi::data_type<Real>(), mpi_shape.x_procs_neg, 200,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(buff.recv_x_neg,
                grid.i_margin * grid.j_total * grid.k_total * n_fields,
                mpi::data_type<Real>(), mpi_shape.x_procs_neg, 100,
                mpi_shape.cart_comm, &reqs[req_count++]);
    }

    // negative y-direction
    if (mpi_shape.y_procs_neg != MPI_PROC_NULL && grid.j_total > 1) {
      pack_y_send<<<cu_shape.grid_dim_y_margin, cu_shape.block_dim>>>(
          buff.view(), qq_trgt.view(), grid.view(), Face::Neg);
      MISO_CUDA_CHECK(cudaGetLastError());
      MISO_CUDA_CHECK(cudaDeviceSynchronize());
      MPI_Isend(buff.send_y_neg,
                grid.i_total * grid.j_margin * grid.k_total * n_fields,
                mpi::data_type<Real>(), mpi_shape.y_procs_neg, 400,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(buff.recv_y_neg,
                grid.i_total * grid.j_margin * grid.k_total * n_fields,
                mpi::data_type<Real>(), mpi_shape.y_procs_neg, 300,
                mpi_shape.cart_comm, &reqs[req_count++]);
    }

    // negative z-direction
    if (mpi_shape.z_procs_neg != MPI_PROC_NULL && grid.k_total > 1) {
      pack_z_send<<<cu_shape.grid_dim_z_margin, cu_shape.block_dim>>>(
          buff.view(), qq_trgt.view(), grid.view(), Face::Neg);
      MISO_CUDA_CHECK(cudaGetLastError());
      MISO_CUDA_CHECK(cudaDeviceSynchronize());
      MPI_Isend(buff.send_z_neg,
                grid.i_total * grid.j_total * grid.k_margin * n_fields,
                mpi::data_type<Real>(), mpi_shape.z_procs_neg, 600,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(buff.recv_z_neg,
                grid.i_total * grid.j_total * grid.k_margin * n_fields,
                mpi::data_type<Real>(), mpi_shape.z_procs_neg, 500,
                mpi_shape.cart_comm, &reqs[req_count++]);
    }

    if (req_count > 0) {
      MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    }

    // ##################
    // positive x-direction
    if (mpi_shape.x_procs_pos != MPI_PROC_NULL && grid.i_total > 1) {
      unpack_x_recv<<<cu_shape.grid_dim_x_margin, cu_shape.block_dim>>>(
          qq_trgt.view(), buff.view(), grid.view(), Face::Pos);
      MISO_CUDA_CHECK(cudaGetLastError());
      MISO_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // positive y-direction
    if (mpi_shape.y_procs_pos != MPI_PROC_NULL && grid.j_total > 1) {
      unpack_y_recv<<<cu_shape.grid_dim_y_margin, cu_shape.block_dim>>>(
          qq_trgt.view(), buff.view(), grid.view(), Face::Pos);
      MISO_CUDA_CHECK(cudaGetLastError());
      MISO_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // positive z-direction
    if (mpi_shape.z_procs_pos != MPI_PROC_NULL && grid.k_total > 1) {
      unpack_z_recv<<<cu_shape.grid_dim_z_margin, cu_shape.block_dim>>>(
          qq_trgt.view(), buff.view(), grid.view(), Face::Pos);
      MISO_CUDA_CHECK(cudaGetLastError());
      MISO_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ##################
    // negative x-direction
    if (mpi_shape.x_procs_neg != MPI_PROC_NULL && grid.i_total > 1) {
      unpack_x_recv<<<cu_shape.grid_dim_x_margin, cu_shape.block_dim>>>(
          qq_trgt.view(), buff.view(), grid.view(), Face::Neg);
      MISO_CUDA_CHECK(cudaGetLastError());
      MISO_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // negative y-direction
    if (mpi_shape.y_procs_neg != MPI_PROC_NULL && grid.j_total > 1) {
      unpack_y_recv<<<cu_shape.grid_dim_y_margin, cu_shape.block_dim>>>(
          qq_trgt.view(), buff.view(), grid.view(), Face::Neg);
      MISO_CUDA_CHECK(cudaGetLastError());
      MISO_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // negative z-direction
    if (mpi_shape.z_procs_neg != MPI_PROC_NULL && grid.k_total > 1) {
      unpack_z_recv<<<cu_shape.grid_dim_z_margin, cu_shape.block_dim>>>(
          qq_trgt.view(), buff.view(), grid.view(), Face::Neg);
      MISO_CUDA_CHECK(cudaGetLastError());
      MISO_CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
};
#endif  // __CUDACC__

}  // namespace mhd
}  // namespace miso
