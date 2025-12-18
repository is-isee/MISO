#pragma once
#include <cassert>

#include <miso/cuda_manager.cuh>
#include <miso/grid_gpu.cuh>
#include <miso/mpi_manager.hpp>

namespace miso {
namespace mhd {

constexpr int n_var = 9;

template <typename Real> struct MHDCore;
template <typename Real> struct MHD;
template <typename Real> struct MHDCoreDevice;

enum class Face { Pos, Neg };

// ##################
// x-direction send
template <typename Real>
__global__ void pack_x_send(MHDCoreDevice<Real> qq_trgt,
                            const GridDevice<Real> grid, Face face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= grid.i_margin || j >= grid.j_total || k >= grid.k_total)
    return;

  Real *__restrict__ send_buff =
      (face == Face::Pos) ? qq_trgt.send_buff_x_pos : qq_trgt.send_buff_x_neg;

  const int i_src = (face == Face::Pos) ? (grid.i_total - 2 * grid.i_margin + i)
                                        : (grid.i_margin + i);
  const size_t src_idx = grid.idx(i_src, j, k);
  const size_t buf_idx = ((i * grid.j_total + j) * grid.k_total + k) * n_var;

  const Real *__restrict__ var[n_var] = {qq_trgt.ro, qq_trgt.vx, qq_trgt.vy,
                                         qq_trgt.vz, qq_trgt.bx, qq_trgt.by,
                                         qq_trgt.bz, qq_trgt.ei, qq_trgt.ph};

#pragma unroll
  for (int m = 0; m < n_var; ++m) {
    send_buff[buf_idx + m] = var[m][src_idx];
  }
};

// y-direction send
template <typename Real>
__global__ void pack_y_send(MHDCoreDevice<Real> qq_trgt,
                            const GridDevice<Real> grid, Face face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= grid.i_total || j >= grid.j_margin || k >= grid.k_total)
    return;

  Real *__restrict__ send_buff =
      (face == Face::Pos) ? qq_trgt.send_buff_y_pos : qq_trgt.send_buff_y_neg;

  const int j_src = (face == Face::Pos) ? (grid.j_total - 2 * grid.j_margin + j)
                                        : (grid.j_margin + j);
  const size_t src_idx = grid.idx(i, j_src, k);
  const size_t buf_idx = ((i * grid.j_margin + j) * grid.k_total + k) * n_var;

  const Real *__restrict__ var[n_var] = {qq_trgt.ro, qq_trgt.vx, qq_trgt.vy,
                                         qq_trgt.vz, qq_trgt.bx, qq_trgt.by,
                                         qq_trgt.bz, qq_trgt.ei, qq_trgt.ph};

#pragma unroll
  for (int m = 0; m < n_var; ++m) {
    send_buff[buf_idx + m] = var[m][src_idx];
  }
};

// z-direction send
template <typename Real>
__global__ void pack_z_send(MHDCoreDevice<Real> qq_trgt,
                            const GridDevice<Real> grid, Face face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= grid.i_total || j >= grid.j_total || k >= grid.k_margin)
    return;

  Real *__restrict__ send_buff =
      (face == Face::Pos) ? qq_trgt.send_buff_z_pos : qq_trgt.send_buff_z_neg;

  const int k_src = (face == Face::Pos) ? (grid.k_total - 2 * grid.k_margin + k)
                                        : (grid.k_margin + k);
  const size_t src_idx = grid.idx(i, j, k_src);
  const size_t buf_idx = ((i * grid.j_total + j) * grid.k_margin + k) * n_var;

  const Real *__restrict__ var[n_var] = {qq_trgt.ro, qq_trgt.vx, qq_trgt.vy,
                                         qq_trgt.vz, qq_trgt.bx, qq_trgt.by,
                                         qq_trgt.bz, qq_trgt.ei, qq_trgt.ph};

#pragma unroll
  for (int m = 0; m < n_var; ++m) {
    send_buff[buf_idx + m] = var[m][src_idx];
  }
};

// ##################
// x-direction recv
template <typename Real>
__global__ void unpack_x_recv(MHDCoreDevice<Real> qq_trgt,
                              const GridDevice<Real> grid, Face face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= grid.i_margin || j >= grid.j_total || k >= grid.k_total)
    return;
  Real *__restrict__ recv_buff =
      (face == Face::Pos) ? qq_trgt.recv_buff_x_pos : qq_trgt.recv_buff_x_neg;
  const int i_tgt = (face == Face::Pos) ? grid.i_total - grid.i_margin + i : i;

  const size_t tgt_idx = grid.idx(i_tgt, j, k);
  const size_t buf_idx = ((i * grid.j_total + j) * grid.k_total + k) * n_var;

  Real *__restrict__ var[n_var] = {qq_trgt.ro, qq_trgt.vx, qq_trgt.vy,
                                   qq_trgt.vz, qq_trgt.bx, qq_trgt.by,
                                   qq_trgt.bz, qq_trgt.ei, qq_trgt.ph};

#pragma unroll
  for (int m = 0; m < n_var; ++m) {
    var[m][tgt_idx] = recv_buff[buf_idx + m];
  }
};

// y-direction recv
template <typename Real>
__global__ void unpack_y_recv(MHDCoreDevice<Real> qq_trgt,
                              const GridDevice<Real> grid, Face face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= grid.i_total || j >= grid.j_margin || k >= grid.k_total)
    return;
  Real *__restrict__ recv_buff =
      (face == Face::Pos) ? qq_trgt.recv_buff_y_pos : qq_trgt.recv_buff_y_neg;
  const int j_tgt = (face == Face::Pos) ? grid.j_total - grid.j_margin + j : j;

  const size_t tgt_idx = grid.idx(i, j_tgt, k);
  const size_t buf_idx = ((i * grid.j_margin + j) * grid.k_total + k) * n_var;

  Real *__restrict__ var[n_var] = {qq_trgt.ro, qq_trgt.vx, qq_trgt.vy,
                                   qq_trgt.vz, qq_trgt.bx, qq_trgt.by,
                                   qq_trgt.bz, qq_trgt.ei, qq_trgt.ph};

#pragma unroll
  for (int m = 0; m < n_var; ++m) {
    var[m][tgt_idx] = recv_buff[buf_idx + m];
  }
};

// z-direction recv
template <typename Real>
__global__ void unpack_z_recv(MHDCoreDevice<Real> qq_trgt,
                              const GridDevice<Real> grid, Face face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= grid.i_total || j >= grid.j_total || k >= grid.k_margin)
    return;
  Real *__restrict__ recv_buff =
      (face == Face::Pos) ? qq_trgt.recv_buff_z_pos : qq_trgt.recv_buff_z_neg;
  const int k_tgt = (face == Face::Pos) ? grid.k_total - grid.k_margin + k : k;

  const size_t tgt_idx = grid.idx(i, j, k_tgt);
  const size_t buf_idx = ((i * grid.j_total + j) * grid.k_margin + k) * n_var;

  Real *__restrict__ var[n_var] = {qq_trgt.ro, qq_trgt.vx, qq_trgt.vy,
                                   qq_trgt.vz, qq_trgt.bx, qq_trgt.by,
                                   qq_trgt.bz, qq_trgt.ei, qq_trgt.ph};

#pragma unroll
  for (int m = 0; m < n_var; ++m) {
    var[m][tgt_idx] = recv_buff[buf_idx + m];
  }
};

/// @brief MHD core data structure on GPU
/// @tparam Real floating point type
template <typename Real> struct MHDCoreDevice {

  Real *ro = nullptr;
  Real *vx = nullptr;
  Real *vy = nullptr;
  Real *vz = nullptr;
  Real *bx = nullptr;
  Real *by = nullptr;
  Real *bz = nullptr;
  Real *ei = nullptr;
  Real *ph = nullptr;

  Real *recv_buff_x_pos = nullptr;
  Real *recv_buff_x_neg = nullptr;
  Real *recv_buff_y_pos = nullptr;
  Real *recv_buff_y_neg = nullptr;
  Real *recv_buff_z_pos = nullptr;
  Real *recv_buff_z_neg = nullptr;
  Real *send_buff_x_pos = nullptr;
  Real *send_buff_x_neg = nullptr;
  Real *send_buff_y_pos = nullptr;
  Real *send_buff_y_neg = nullptr;
  Real *send_buff_z_pos = nullptr;
  Real *send_buff_z_neg = nullptr;

  int i_total, j_total, k_total;

  // clang-format off
    MHDCoreDevice(const Grid<Real> &grid)
        : i_total(grid.i_total), j_total(grid.j_total), k_total(grid.k_total) {
      CUDA_CHECK(cudaMalloc(&ro, sizeof(Real) * i_total * j_total * k_total));
      CUDA_CHECK(cudaMalloc(&vx, sizeof(Real) * i_total * j_total * k_total));
      CUDA_CHECK(cudaMalloc(&vy, sizeof(Real) * i_total * j_total * k_total));
      CUDA_CHECK(cudaMalloc(&vz, sizeof(Real) * i_total * j_total * k_total));
      CUDA_CHECK(cudaMalloc(&bx, sizeof(Real) * i_total * j_total * k_total));
      CUDA_CHECK(cudaMalloc(&by, sizeof(Real) * i_total * j_total * k_total));
      CUDA_CHECK(cudaMalloc(&bz, sizeof(Real) * i_total * j_total * k_total));
      CUDA_CHECK(cudaMalloc(&ei, sizeof(Real) * i_total * j_total * k_total));
      CUDA_CHECK(cudaMalloc(&ph, sizeof(Real) * i_total * j_total * k_total));
      CUDA_CHECK(cudaMalloc(&recv_buff_x_pos, sizeof(Real) * grid.i_margin * j_total * k_total * n_var));
      CUDA_CHECK(cudaMalloc(&recv_buff_x_neg, sizeof(Real) * grid.i_margin * j_total * k_total * n_var));
      CUDA_CHECK(cudaMalloc(&recv_buff_y_pos, sizeof(Real) * grid.j_margin * i_total * k_total * n_var));
      CUDA_CHECK(cudaMalloc(&recv_buff_y_neg, sizeof(Real) * grid.j_margin * i_total * k_total * n_var));
      CUDA_CHECK(cudaMalloc(&recv_buff_z_pos, sizeof(Real) * grid.k_margin * i_total * j_total * n_var));
      CUDA_CHECK(cudaMalloc(&recv_buff_z_neg, sizeof(Real) * grid.k_margin * i_total * j_total * n_var));
      CUDA_CHECK(cudaMalloc(&send_buff_x_pos, sizeof(Real) * grid.i_margin * j_total * k_total * n_var));
      CUDA_CHECK(cudaMalloc(&send_buff_x_neg, sizeof(Real) * grid.i_margin * j_total * k_total * n_var));
      CUDA_CHECK(cudaMalloc(&send_buff_y_pos, sizeof(Real) * grid.j_margin * i_total * k_total * n_var));
      CUDA_CHECK(cudaMalloc(&send_buff_y_neg, sizeof(Real) * grid.j_margin * i_total * k_total * n_var));
      CUDA_CHECK(cudaMalloc(&send_buff_z_pos, sizeof(Real) * grid.k_margin * i_total * j_total * n_var));
      CUDA_CHECK(cudaMalloc(&send_buff_z_neg, sizeof(Real) * grid.k_margin * i_total * j_total * n_var));
    }
  // clang-format on

  ~MHDCoreDevice() {}

  void free() {
    auto F = [](Real *&p) {
      if (p)
        CUDA_CHECK(cudaFree(p));
      p = nullptr;
    };
    // clang-format off
      // fields
      F(ro); F(vx); F(vy); F(vz); F(bx); F(by); F(bz); F(ei); F(ph);

      // buffers
      F(recv_buff_x_pos); F(recv_buff_x_neg);
      F(recv_buff_y_pos); F(recv_buff_y_neg);
      F(recv_buff_z_pos); F(recv_buff_z_neg);
      F(send_buff_x_pos); F(send_buff_x_neg);
      F(send_buff_y_pos); F(send_buff_y_neg);
      F(send_buff_z_pos); F(send_buff_z_neg);
    // clang-format on
  }

  MHDCoreDevice(const MHDCoreDevice &) = default;
  MHDCoreDevice &operator=(const MHDCoreDevice &) = default;

  void copy_from_host(const MHDCore<Real> &qq_h, CudaManager<Real> &cuda) {
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaMemcpyAsync(ro, qq_h.ro.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_ro));
    CUDA_CHECK(cudaMemcpyAsync(vx, qq_h.vx.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_vx));
    CUDA_CHECK(cudaMemcpyAsync(vy, qq_h.vy.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_vy));
    CUDA_CHECK(cudaMemcpyAsync(vz, qq_h.vz.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_vz));
    CUDA_CHECK(cudaMemcpyAsync(bx, qq_h.bx.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_bx));
    CUDA_CHECK(cudaMemcpyAsync(by, qq_h.by.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_by));
    CUDA_CHECK(cudaMemcpyAsync(bz, qq_h.bz.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_bz));
    CUDA_CHECK(cudaMemcpyAsync(ei, qq_h.ei.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_ei));
    CUDA_CHECK(cudaMemcpyAsync(ph, qq_h.ph.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_ph));

    cudaDeviceSynchronize();
  }

  void copy_to_host(MHDCore<Real> &qq_h, CudaManager<Real> &cuda) {
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaMemcpyAsync(qq_h.ro.data(), ro,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_ro));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.vx.data(), vx,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_vx));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.vy.data(), vy,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_vy));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.vz.data(), vz,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_vz));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.bx.data(), bx,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_bx));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.by.data(), by,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_by));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.bz.data(), bz,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_bz));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.ei.data(), ei,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_ei));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.ph.data(), ph,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_ph));

    cudaDeviceSynchronize();
  }

  void copy_from_device(const MHDCoreDevice<Real> &qq_d,
                        CudaManager<Real> &cuda) {
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaMemcpyAsync(ro, qq_d.ro,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_ro));
    CUDA_CHECK(cudaMemcpyAsync(vx, qq_d.vx,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_vx));
    CUDA_CHECK(cudaMemcpyAsync(vy, qq_d.vy,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_vy));
    CUDA_CHECK(cudaMemcpyAsync(vz, qq_d.vz,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_vz));
    CUDA_CHECK(cudaMemcpyAsync(bx, qq_d.bx,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_bx));
    CUDA_CHECK(cudaMemcpyAsync(by, qq_d.by,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_by));
    CUDA_CHECK(cudaMemcpyAsync(bz, qq_d.bz,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_bz));
    CUDA_CHECK(cudaMemcpyAsync(ei, qq_d.ei,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_ei));
    CUDA_CHECK(cudaMemcpyAsync(ph, qq_d.ph,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_ph));

    cudaDeviceSynchronize();
  }
};

template <typename Real> struct MHDDevice {
  MHDCoreDevice<Real> qq, qq_argm, qq_rslt;
  Real cfl_number;

  MHDDevice(const Grid<Real> &grid, const MHD<Real> &mhd)
      : cfl_number(mhd.cfl_number), qq(grid), qq_argm(grid), qq_rslt(grid) {}

  // destructor (by default GPU memory is freed when MHDDevice is destroyed)
  ~MHDDevice() {}

  void free() {
    qq.free();
    qq_argm.free();
    qq_rslt.free();
  }

  MHDDevice(const MHDDevice &) = default;
  MHDDevice &operator=(const MHDDevice &) = default;

  void mpi_exchange_halo(MHDCoreDevice<Real> &qq_trgt, GridDevice<Real> &grid,
                         MPIManager &mpi, CudaManager<Real> &cuda) {
    MPI_Request reqs[12];
    int req_count = 0;

    // ##################
    // positive x-direction
    if (mpi.x_procs_pos != MPI_PROC_NULL && grid.i_total > 1) {
      pack_x_send<<<cuda.grid_dim_x_margin, cuda.block_dim>>>(qq_trgt, grid,
                                                              Face::Pos);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      MPI_Isend(qq_trgt.send_buff_x_pos,
                grid.i_margin * grid.j_total * grid.k_total * n_var,
                mpi_type<Real>(), mpi.x_procs_pos, 100, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(qq_trgt.recv_buff_x_pos,
                grid.i_margin * grid.j_total * grid.k_total * n_var,
                mpi_type<Real>(), mpi.x_procs_pos, 200, mpi.cart_comm,
                &reqs[req_count++]);
    }

    // positive y-direction
    if (mpi.y_procs_pos != MPI_PROC_NULL && grid.j_total > 1) {
      pack_y_send<<<cuda.grid_dim_y_margin, cuda.block_dim>>>(qq_trgt, grid,
                                                              Face::Pos);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      MPI_Isend(qq_trgt.send_buff_y_pos,
                grid.i_total * grid.j_margin * grid.k_total * n_var,
                mpi_type<Real>(), mpi.y_procs_pos, 300, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(qq_trgt.recv_buff_y_pos,
                grid.i_total * grid.j_margin * grid.k_total * n_var,
                mpi_type<Real>(), mpi.y_procs_pos, 400, mpi.cart_comm,
                &reqs[req_count++]);
    }

    // positive z-direction
    if (mpi.z_procs_pos != MPI_PROC_NULL && grid.k_total > 1) {
      pack_z_send<<<cuda.grid_dim_z_margin, cuda.block_dim>>>(qq_trgt, grid,
                                                              Face::Pos);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      MPI_Isend(qq_trgt.send_buff_z_pos,
                grid.i_total * grid.j_total * grid.k_margin * n_var,
                mpi_type<Real>(), mpi.z_procs_pos, 500, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(qq_trgt.recv_buff_z_pos,
                grid.i_total * grid.j_total * grid.k_margin * n_var,
                mpi_type<Real>(), mpi.z_procs_pos, 600, mpi.cart_comm,
                &reqs[req_count++]);
    }

    // ##################
    // negative x-direction
    if (mpi.x_procs_neg != MPI_PROC_NULL && grid.i_total > 1) {
      pack_x_send<<<cuda.grid_dim_x_margin, cuda.block_dim>>>(qq_trgt, grid,
                                                              Face::Neg);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      MPI_Isend(qq_trgt.send_buff_x_neg,
                grid.i_margin * grid.j_total * grid.k_total * n_var,
                mpi_type<Real>(), mpi.x_procs_neg, 200, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(qq_trgt.recv_buff_x_neg,
                grid.i_margin * grid.j_total * grid.k_total * n_var,
                mpi_type<Real>(), mpi.x_procs_neg, 100, mpi.cart_comm,
                &reqs[req_count++]);
    }

    // negative y-direction
    if (mpi.y_procs_neg != MPI_PROC_NULL && grid.j_total > 1) {
      pack_y_send<<<cuda.grid_dim_y_margin, cuda.block_dim>>>(qq_trgt, grid,
                                                              Face::Neg);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      MPI_Isend(qq_trgt.send_buff_y_neg,
                grid.i_total * grid.j_margin * grid.k_total * n_var,
                mpi_type<Real>(), mpi.y_procs_neg, 400, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(qq_trgt.recv_buff_y_neg,
                grid.i_total * grid.j_margin * grid.k_total * n_var,
                mpi_type<Real>(), mpi.y_procs_neg, 300, mpi.cart_comm,
                &reqs[req_count++]);
    }

    // negative z-direction
    if (mpi.z_procs_neg != MPI_PROC_NULL && grid.k_total > 1) {
      pack_z_send<<<cuda.grid_dim_z_margin, cuda.block_dim>>>(qq_trgt, grid,
                                                              Face::Neg);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      MPI_Isend(qq_trgt.send_buff_z_neg,
                grid.i_total * grid.j_total * grid.k_margin * n_var,
                mpi_type<Real>(), mpi.z_procs_neg, 600, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(qq_trgt.recv_buff_z_neg,
                grid.i_total * grid.j_total * grid.k_margin * n_var,
                mpi_type<Real>(), mpi.z_procs_neg, 500, mpi.cart_comm,
                &reqs[req_count++]);
    }

    if (req_count > 0) {
      MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    }

    // ##################
    // positive x-direction
    if (mpi.x_procs_pos != MPI_PROC_NULL && grid.i_total > 1) {
      unpack_x_recv<<<cuda.grid_dim_x_margin, cuda.block_dim>>>(qq_trgt, grid,
                                                                Face::Pos);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // positive y-direction
    if (mpi.y_procs_pos != MPI_PROC_NULL && grid.j_total > 1) {
      unpack_y_recv<<<cuda.grid_dim_y_margin, cuda.block_dim>>>(qq_trgt, grid,
                                                                Face::Pos);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // positive z-direction
    if (mpi.z_procs_pos != MPI_PROC_NULL && grid.k_total > 1) {
      unpack_z_recv<<<cuda.grid_dim_z_margin, cuda.block_dim>>>(qq_trgt, grid,
                                                                Face::Pos);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ##################
    // negative x-direction
    if (mpi.x_procs_neg != MPI_PROC_NULL && grid.i_total > 1) {
      unpack_x_recv<<<cuda.grid_dim_x_margin, cuda.block_dim>>>(qq_trgt, grid,
                                                                Face::Neg);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // negative y-direction
    if (mpi.y_procs_neg != MPI_PROC_NULL && grid.j_total > 1) {
      unpack_y_recv<<<cuda.grid_dim_y_margin, cuda.block_dim>>>(qq_trgt, grid,
                                                                Face::Neg);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // negative z-direction
    if (mpi.z_procs_neg != MPI_PROC_NULL && grid.k_total > 1) {
      unpack_z_recv<<<cuda.grid_dim_z_margin, cuda.block_dim>>>(qq_trgt, grid,
                                                                Face::Neg);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
};

}  // namespace mhd
}  // namespace miso
