#pragma once
#include <cassert>

#include <miso/cuda_utils.cuh>
#include <miso/grid_gpu.cuh>
#include <miso/mpi_manager.hpp>

namespace miso {
namespace mhd {

constexpr int n_var = 9;

template <typename Real> struct MHDCore;
template <typename Real> struct MHD;
template <typename Real> struct MHDCoreDevice;

enum class Face { Pos, Neg };

struct MHDStreams {
  cudaStream_t stream_ro;
  cudaStream_t stream_vx;
  cudaStream_t stream_vy;
  cudaStream_t stream_vz;
  cudaStream_t stream_bx;
  cudaStream_t stream_by;
  cudaStream_t stream_bz;
  cudaStream_t stream_ei;
  cudaStream_t stream_ph;

  MHDStreams() {
    CUDA_CHECK(cudaStreamCreate(&stream_ro));
    CUDA_CHECK(cudaStreamCreate(&stream_vx));
    CUDA_CHECK(cudaStreamCreate(&stream_vy));
    CUDA_CHECK(cudaStreamCreate(&stream_vz));
    CUDA_CHECK(cudaStreamCreate(&stream_bx));
    CUDA_CHECK(cudaStreamCreate(&stream_by));
    CUDA_CHECK(cudaStreamCreate(&stream_bz));
    CUDA_CHECK(cudaStreamCreate(&stream_ei));
    CUDA_CHECK(cudaStreamCreate(&stream_ph));
  }

  ~MHDStreams() {
    if (stream_ro)
      cudaStreamDestroy(stream_ro);
    if (stream_vx)
      cudaStreamDestroy(stream_vx);
    if (stream_vy)
      cudaStreamDestroy(stream_vy);
    if (stream_vz)
      cudaStreamDestroy(stream_vz);
    if (stream_bx)
      cudaStreamDestroy(stream_bx);
    if (stream_by)
      cudaStreamDestroy(stream_by);
    if (stream_bz)
      cudaStreamDestroy(stream_bz);
    if (stream_ei)
      cudaStreamDestroy(stream_ei);
    if (stream_ph)
      cudaStreamDestroy(stream_ph);
  }

  auto ro() const noexcept { return stream_ro; }
  auto vx() const noexcept { return stream_vx; }
  auto vy() const noexcept { return stream_vy; }
  auto vz() const noexcept { return stream_vz; }
  auto bx() const noexcept { return stream_bx; }
  auto by() const noexcept { return stream_by; }
  auto bz() const noexcept { return stream_bz; }
  auto ei() const noexcept { return stream_ei; }
  auto ph() const noexcept { return stream_ph; }

  // Prohibit copy and move operations
  MHDStreams(const MHDStreams &) = delete;
  MHDStreams &operator=(const MHDStreams &) = delete;
  MHDStreams(MHDStreams &&) = delete;
  MHDStreams &operator=(MHDStreams &&) = delete;
};

/// @brief MHD variables on GPU
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
  size_t array_size;

  MHDCoreDevice(const Grid<Real> &grid) {
    array_size = sizeof(Real) * grid.i_total * grid.j_total * grid.k_total;
    CUDA_CHECK(cudaMalloc(&ro, array_size));
    CUDA_CHECK(cudaMalloc(&vx, array_size));
    CUDA_CHECK(cudaMalloc(&vy, array_size));
    CUDA_CHECK(cudaMalloc(&vz, array_size));
    CUDA_CHECK(cudaMalloc(&bx, array_size));
    CUDA_CHECK(cudaMalloc(&by, array_size));
    CUDA_CHECK(cudaMalloc(&bz, array_size));
    CUDA_CHECK(cudaMalloc(&ei, array_size));
    CUDA_CHECK(cudaMalloc(&ph, array_size));
  }

  ~MHDCoreDevice() {}

  void free() {
    auto F = [](Real *&p) {
      if (p)
        CUDA_CHECK(cudaFree(p));
      p = nullptr;
    };
    // clang-format off
    F(ro); F(vx); F(vy); F(vz); F(bx); F(by); F(bz); F(ei); F(ph);
    // clang-format on
  }

  // Allow default copy constructor and copy assignment
  MHDCoreDevice(const MHDCoreDevice &) = default;
  MHDCoreDevice &operator=(const MHDCoreDevice &) = default;

  void copy_from_host(const MHDCore<Real> &qq_h, MHDStreams &mhd_streams) {
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpyAsync(ro, qq_h.ro.data(), array_size,
                               cudaMemcpyHostToDevice, mhd_streams.ro()));
    CUDA_CHECK(cudaMemcpyAsync(vx, qq_h.vx.data(), array_size,
                               cudaMemcpyHostToDevice, mhd_streams.vx()));
    CUDA_CHECK(cudaMemcpyAsync(vy, qq_h.vy.data(), array_size,
                               cudaMemcpyHostToDevice, mhd_streams.vy()));
    CUDA_CHECK(cudaMemcpyAsync(vz, qq_h.vz.data(), array_size,
                               cudaMemcpyHostToDevice, mhd_streams.vz()));
    CUDA_CHECK(cudaMemcpyAsync(bx, qq_h.bx.data(), array_size,
                               cudaMemcpyHostToDevice, mhd_streams.bx()));
    CUDA_CHECK(cudaMemcpyAsync(by, qq_h.by.data(), array_size,
                               cudaMemcpyHostToDevice, mhd_streams.by()));
    CUDA_CHECK(cudaMemcpyAsync(bz, qq_h.bz.data(), array_size,
                               cudaMemcpyHostToDevice, mhd_streams.bz()));
    CUDA_CHECK(cudaMemcpyAsync(ei, qq_h.ei.data(), array_size,
                               cudaMemcpyHostToDevice, mhd_streams.ei()));
    CUDA_CHECK(cudaMemcpyAsync(ph, qq_h.ph.data(), array_size,
                               cudaMemcpyHostToDevice, mhd_streams.ph()));
    cudaDeviceSynchronize();
  }

  void copy_to_host(MHDCore<Real> &qq_h, MHDStreams &mhd_streams) {
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpyAsync(qq_h.ro.data(), ro, array_size,
                               cudaMemcpyDeviceToHost, mhd_streams.ro()));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.vx.data(), vx, array_size,
                               cudaMemcpyDeviceToHost, mhd_streams.vx()));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.vy.data(), vy, array_size,
                               cudaMemcpyDeviceToHost, mhd_streams.vy()));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.vz.data(), vz, array_size,
                               cudaMemcpyDeviceToHost, mhd_streams.vz()));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.bx.data(), bx, array_size,
                               cudaMemcpyDeviceToHost, mhd_streams.bx()));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.by.data(), by, array_size,
                               cudaMemcpyDeviceToHost, mhd_streams.by()));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.bz.data(), bz, array_size,
                               cudaMemcpyDeviceToHost, mhd_streams.bz()));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.ei.data(), ei, array_size,
                               cudaMemcpyDeviceToHost, mhd_streams.ei()));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.ph.data(), ph, array_size,
                               cudaMemcpyDeviceToHost, mhd_streams.ph()));
    cudaDeviceSynchronize();
  }

  void copy_from_device(const MHDCoreDevice<Real> &qq_d,
                        MHDStreams &mhd_streams) {
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpyAsync(ro, qq_d.ro, array_size, cudaMemcpyDeviceToDevice,
                               mhd_streams.ro()));
    CUDA_CHECK(cudaMemcpyAsync(vx, qq_d.vx, array_size, cudaMemcpyDeviceToDevice,
                               mhd_streams.vx()));
    CUDA_CHECK(cudaMemcpyAsync(vy, qq_d.vy, array_size, cudaMemcpyDeviceToDevice,
                               mhd_streams.vy()));
    CUDA_CHECK(cudaMemcpyAsync(vz, qq_d.vz, array_size, cudaMemcpyDeviceToDevice,
                               mhd_streams.vz()));
    CUDA_CHECK(cudaMemcpyAsync(bx, qq_d.bx, array_size, cudaMemcpyDeviceToDevice,
                               mhd_streams.bx()));
    CUDA_CHECK(cudaMemcpyAsync(by, qq_d.by, array_size, cudaMemcpyDeviceToDevice,
                               mhd_streams.by()));
    CUDA_CHECK(cudaMemcpyAsync(bz, qq_d.bz, array_size, cudaMemcpyDeviceToDevice,
                               mhd_streams.bz()));
    CUDA_CHECK(cudaMemcpyAsync(ei, qq_d.ei, array_size, cudaMemcpyDeviceToDevice,
                               mhd_streams.ei()));
    CUDA_CHECK(cudaMemcpyAsync(ph, qq_d.ph, array_size, cudaMemcpyDeviceToDevice,
                               mhd_streams.ph()));
    cudaDeviceSynchronize();
  }
};

/// @brief MHD buffers on GPU
template <typename Real> struct MHDBufferDevice {
  Real *recv_x_pos = nullptr;
  Real *recv_x_neg = nullptr;
  Real *recv_y_pos = nullptr;
  Real *recv_y_neg = nullptr;
  Real *recv_z_pos = nullptr;
  Real *recv_z_neg = nullptr;
  Real *send_x_pos = nullptr;
  Real *send_x_neg = nullptr;
  Real *send_y_pos = nullptr;
  Real *send_y_neg = nullptr;
  Real *send_z_pos = nullptr;
  Real *send_z_neg = nullptr;

  MHDBufferDevice(const Grid<Real> &grid) {
    const auto buff_size_x =
        sizeof(Real) * grid.i_margin * grid.j_total * grid.k_total * n_var;
    const auto buff_size_y =
        sizeof(Real) * grid.j_margin * grid.i_total * grid.k_total * n_var;
    const auto buff_size_z =
        sizeof(Real) * grid.k_margin * grid.i_total * grid.j_total * n_var;
    CUDA_CHECK(cudaMalloc(&recv_x_pos, buff_size_x));
    CUDA_CHECK(cudaMalloc(&recv_x_neg, buff_size_x));
    CUDA_CHECK(cudaMalloc(&recv_y_pos, buff_size_y));
    CUDA_CHECK(cudaMalloc(&recv_y_neg, buff_size_y));
    CUDA_CHECK(cudaMalloc(&recv_z_pos, buff_size_z));
    CUDA_CHECK(cudaMalloc(&recv_z_neg, buff_size_z));
    CUDA_CHECK(cudaMalloc(&send_x_pos, buff_size_x));
    CUDA_CHECK(cudaMalloc(&send_x_neg, buff_size_x));
    CUDA_CHECK(cudaMalloc(&send_y_pos, buff_size_y));
    CUDA_CHECK(cudaMalloc(&send_y_neg, buff_size_y));
    CUDA_CHECK(cudaMalloc(&send_z_pos, buff_size_z));
    CUDA_CHECK(cudaMalloc(&send_z_neg, buff_size_z));
  }

  ~MHDBufferDevice() {}

  void free() {
    auto F = [](Real *&p) {
      if (p)
        CUDA_CHECK(cudaFree(p));
      p = nullptr;
    };
    // clang-format off
    F(recv_x_pos); F(recv_x_neg);
    F(recv_y_pos); F(recv_y_neg);
    F(recv_z_pos); F(recv_z_neg);
    F(send_x_pos); F(send_x_neg);
    F(send_y_pos); F(send_y_neg);
    F(send_z_pos); F(send_z_neg);
    // clang-format on
  }

  // Allow default copy constructor and copy assignment
  MHDBufferDevice(const MHDBufferDevice &) = default;
  MHDBufferDevice &operator=(const MHDBufferDevice &) = default;
};

// ##################
// x-direction send
template <typename Real>
__global__ void pack_x_send(MHDBufferDevice<Real> buff,
                            const MHDCoreDevice<Real> qq_trgt,
                            const GridDevice<Real> grid, Face face) {
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
__global__ void pack_y_send(MHDBufferDevice<Real> buff,
                            const MHDCoreDevice<Real> qq_trgt,
                            const GridDevice<Real> grid, Face face) {
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
__global__ void pack_z_send(MHDBufferDevice<Real> buff,
                            const MHDCoreDevice<Real> qq_trgt,
                            const GridDevice<Real> grid, Face face) {
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
                              const MHDBufferDevice<Real> buff,
                              const GridDevice<Real> grid, Face face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= grid.i_margin || j >= grid.j_total || k >= grid.k_total)
    return;
  Real *__restrict__ recv_buff =
      (face == Face::Pos) ? buff.recv_x_pos : buff.recv_x_neg;
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
                              const MHDBufferDevice<Real> buff,
                              const GridDevice<Real> grid, Face face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= grid.i_total || j >= grid.j_margin || k >= grid.k_total)
    return;
  Real *__restrict__ recv_buff =
      (face == Face::Pos) ? buff.recv_y_pos : buff.recv_y_neg;
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
                              const MHDBufferDevice<Real> buff,
                              const GridDevice<Real> grid, Face face) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= grid.i_total || j >= grid.j_total || k >= grid.k_margin)
    return;
  Real *__restrict__ recv_buff =
      (face == Face::Pos) ? buff.recv_z_pos : buff.recv_z_neg;
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

template <typename Real> struct MHDDevice {
  MHDCoreDevice<Real> qq, qq_argm, qq_rslt;
  MHDBufferDevice<Real> buff;
  Real cfl_number;

  MHDDevice(const Grid<Real> &grid, const MHD<Real> &mhd)
      : cfl_number(mhd.cfl_number), qq(grid), qq_argm(grid), qq_rslt(grid),
        buff(grid) {}

  // destructor (by default GPU memory is freed when MHDDevice is destroyed)
  ~MHDDevice() {}

  void free() {
    qq.free();
    qq_argm.free();
    qq_rslt.free();
    buff.free();
  }

  MHDDevice(const MHDDevice &) = default;
  MHDDevice &operator=(const MHDDevice &) = default;

  void mpi_exchange_halo(MHDCoreDevice<Real> &qq_trgt, GridDevice<Real> &grid,
                         MPIManager &mpi, CudaKernelShape<Real> &cu_shape) {
    MPI_Request reqs[12];
    int req_count = 0;

    // ##################
    // positive x-direction
    if (mpi.x_procs_pos != MPI_PROC_NULL && grid.i_total > 1) {
      pack_x_send<<<cu_shape.grid_dim_x_margin, cu_shape.block_dim>>>(
          buff, qq_trgt, grid, Face::Pos);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      MPI_Isend(buff.send_x_pos,
                grid.i_margin * grid.j_total * grid.k_total * n_var,
                mpi_type<Real>(), mpi.x_procs_pos, 100, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(buff.recv_x_pos,
                grid.i_margin * grid.j_total * grid.k_total * n_var,
                mpi_type<Real>(), mpi.x_procs_pos, 200, mpi.cart_comm,
                &reqs[req_count++]);
    }

    // positive y-direction
    if (mpi.y_procs_pos != MPI_PROC_NULL && grid.j_total > 1) {
      pack_y_send<<<cu_shape.grid_dim_y_margin, cu_shape.block_dim>>>(
          buff, qq_trgt, grid, Face::Pos);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      MPI_Isend(buff.send_y_pos,
                grid.i_total * grid.j_margin * grid.k_total * n_var,
                mpi_type<Real>(), mpi.y_procs_pos, 300, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(buff.recv_y_pos,
                grid.i_total * grid.j_margin * grid.k_total * n_var,
                mpi_type<Real>(), mpi.y_procs_pos, 400, mpi.cart_comm,
                &reqs[req_count++]);
    }

    // positive z-direction
    if (mpi.z_procs_pos != MPI_PROC_NULL && grid.k_total > 1) {
      pack_z_send<<<cu_shape.grid_dim_z_margin, cu_shape.block_dim>>>(
          buff, qq_trgt, grid, Face::Pos);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      MPI_Isend(buff.send_z_pos,
                grid.i_total * grid.j_total * grid.k_margin * n_var,
                mpi_type<Real>(), mpi.z_procs_pos, 500, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(buff.recv_z_pos,
                grid.i_total * grid.j_total * grid.k_margin * n_var,
                mpi_type<Real>(), mpi.z_procs_pos, 600, mpi.cart_comm,
                &reqs[req_count++]);
    }

    // ##################
    // negative x-direction
    if (mpi.x_procs_neg != MPI_PROC_NULL && grid.i_total > 1) {
      pack_x_send<<<cu_shape.grid_dim_x_margin, cu_shape.block_dim>>>(
          buff, qq_trgt, grid, Face::Neg);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      MPI_Isend(buff.send_x_neg,
                grid.i_margin * grid.j_total * grid.k_total * n_var,
                mpi_type<Real>(), mpi.x_procs_neg, 200, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(buff.recv_x_neg,
                grid.i_margin * grid.j_total * grid.k_total * n_var,
                mpi_type<Real>(), mpi.x_procs_neg, 100, mpi.cart_comm,
                &reqs[req_count++]);
    }

    // negative y-direction
    if (mpi.y_procs_neg != MPI_PROC_NULL && grid.j_total > 1) {
      pack_y_send<<<cu_shape.grid_dim_y_margin, cu_shape.block_dim>>>(
          buff, qq_trgt, grid, Face::Neg);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      MPI_Isend(buff.send_y_neg,
                grid.i_total * grid.j_margin * grid.k_total * n_var,
                mpi_type<Real>(), mpi.y_procs_neg, 400, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(buff.recv_y_neg,
                grid.i_total * grid.j_margin * grid.k_total * n_var,
                mpi_type<Real>(), mpi.y_procs_neg, 300, mpi.cart_comm,
                &reqs[req_count++]);
    }

    // negative z-direction
    if (mpi.z_procs_neg != MPI_PROC_NULL && grid.k_total > 1) {
      pack_z_send<<<cu_shape.grid_dim_z_margin, cu_shape.block_dim>>>(
          buff, qq_trgt, grid, Face::Neg);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      MPI_Isend(buff.send_z_neg,
                grid.i_total * grid.j_total * grid.k_margin * n_var,
                mpi_type<Real>(), mpi.z_procs_neg, 600, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(buff.recv_z_neg,
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
      unpack_x_recv<<<cu_shape.grid_dim_x_margin, cu_shape.block_dim>>>(
          qq_trgt, buff, grid, Face::Pos);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // positive y-direction
    if (mpi.y_procs_pos != MPI_PROC_NULL && grid.j_total > 1) {
      unpack_y_recv<<<cu_shape.grid_dim_y_margin, cu_shape.block_dim>>>(
          qq_trgt, buff, grid, Face::Pos);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // positive z-direction
    if (mpi.z_procs_pos != MPI_PROC_NULL && grid.k_total > 1) {
      unpack_z_recv<<<cu_shape.grid_dim_z_margin, cu_shape.block_dim>>>(
          qq_trgt, buff, grid, Face::Pos);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ##################
    // negative x-direction
    if (mpi.x_procs_neg != MPI_PROC_NULL && grid.i_total > 1) {
      unpack_x_recv<<<cu_shape.grid_dim_x_margin, cu_shape.block_dim>>>(
          qq_trgt, buff, grid, Face::Neg);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // negative y-direction
    if (mpi.y_procs_neg != MPI_PROC_NULL && grid.j_total > 1) {
      unpack_y_recv<<<cu_shape.grid_dim_y_margin, cu_shape.block_dim>>>(
          qq_trgt, buff, grid, Face::Neg);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // negative z-direction
    if (mpi.z_procs_neg != MPI_PROC_NULL && grid.k_total > 1) {
      unpack_z_recv<<<cu_shape.grid_dim_z_margin, cu_shape.block_dim>>>(
          qq_trgt, buff, grid, Face::Neg);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
};

}  // namespace mhd
}  // namespace miso
