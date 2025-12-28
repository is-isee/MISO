#pragma once
#include <cassert>

#include <miso/cuda_util.cuh>
#include <miso/grid_gpu.cuh>
#include <miso/mhd_view.hpp>
#include <miso/mpi_util.hpp>

namespace miso {
namespace mhd {

namespace impl_host {
template <typename Real> struct Fields;
}  // namespace impl_host

namespace impl_cuda {

constexpr int n_vars = 9;

enum class Face { Pos, Neg };

struct Streams {
  cudaStream_t stream_ro;
  cudaStream_t stream_vx;
  cudaStream_t stream_vy;
  cudaStream_t stream_vz;
  cudaStream_t stream_bx;
  cudaStream_t stream_by;
  cudaStream_t stream_bz;
  cudaStream_t stream_ei;
  cudaStream_t stream_ph;

  Streams() {
    MISO_CUDA_CHECK(cudaStreamCreate(&stream_ro));
    MISO_CUDA_CHECK(cudaStreamCreate(&stream_vx));
    MISO_CUDA_CHECK(cudaStreamCreate(&stream_vy));
    MISO_CUDA_CHECK(cudaStreamCreate(&stream_vz));
    MISO_CUDA_CHECK(cudaStreamCreate(&stream_bx));
    MISO_CUDA_CHECK(cudaStreamCreate(&stream_by));
    MISO_CUDA_CHECK(cudaStreamCreate(&stream_bz));
    MISO_CUDA_CHECK(cudaStreamCreate(&stream_ei));
    MISO_CUDA_CHECK(cudaStreamCreate(&stream_ph));
  }

  ~Streams() {
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
  Streams(const Streams &) = delete;
  Streams &operator=(const Streams &) = delete;
  Streams(Streams &&) = delete;
  Streams &operator=(Streams &&) = delete;
};

/// @brief Execution context for MHD on GPU
struct ExecContext {
  mpi::Shape &mpi_shape;
  cuda::KernelShape3D &cu_shape;
  Streams &mhd_streams;
};

/// @brief MHD variables on GPU
template <typename Real> struct Fields {
  Real *ro = nullptr;
  Real *vx = nullptr;
  Real *vy = nullptr;
  Real *vz = nullptr;
  Real *bx = nullptr;
  Real *by = nullptr;
  Real *bz = nullptr;
  Real *ei = nullptr;
  Real *ph = nullptr;
  int i_total = -1, j_total = -1, k_total = -1;
  size_t array_size = 0;

  Fields(const GridDevice<Real> &grid)
      : i_total(grid.i_total), j_total(grid.j_total), k_total(grid.k_total) {
    array_size = sizeof(Real) * i_total * j_total * k_total;
    MISO_CUDA_CHECK(cudaMalloc(&ro, array_size));
    MISO_CUDA_CHECK(cudaMalloc(&vx, array_size));
    MISO_CUDA_CHECK(cudaMalloc(&vy, array_size));
    MISO_CUDA_CHECK(cudaMalloc(&vz, array_size));
    MISO_CUDA_CHECK(cudaMalloc(&bx, array_size));
    MISO_CUDA_CHECK(cudaMalloc(&by, array_size));
    MISO_CUDA_CHECK(cudaMalloc(&bz, array_size));
    MISO_CUDA_CHECK(cudaMalloc(&ei, array_size));
    MISO_CUDA_CHECK(cudaMalloc(&ph, array_size));
  }

  ~Fields() {
    auto F = [](Real *&p) {
      if (p)
        MISO_CUDA_CHECK(cudaFree(p));
      p = nullptr;
    };
    // clang-format off
    F(ro); F(vx); F(vy); F(vz);
    F(bx); F(by); F(bz); F(ei); F(ph);
    // clang-format on
  }

  // Shallow-const / shallow-copy
  FieldsView<Real> view() const noexcept { return FieldsView<Real>(*this); }

  // Prohibit copy and move operations
  Fields(const Fields &) = delete;
  Fields &operator=(const Fields &) = delete;
  Fields(Fields &&) = delete;
  Fields &operator=(Fields &&) = delete;

  void copy_from_host(const impl_host::Fields<Real> &qq_h, Streams &cu_streams) {
    cudaDeviceSynchronize();
    MISO_CUDA_CHECK(cudaMemcpyAsync(ro, qq_h.ro.data(), array_size,
                                    cudaMemcpyHostToDevice, cu_streams.ro()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(vx, qq_h.vx.data(), array_size,
                                    cudaMemcpyHostToDevice, cu_streams.vx()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(vy, qq_h.vy.data(), array_size,
                                    cudaMemcpyHostToDevice, cu_streams.vy()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(vz, qq_h.vz.data(), array_size,
                                    cudaMemcpyHostToDevice, cu_streams.vz()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(bx, qq_h.bx.data(), array_size,
                                    cudaMemcpyHostToDevice, cu_streams.bx()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(by, qq_h.by.data(), array_size,
                                    cudaMemcpyHostToDevice, cu_streams.by()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(bz, qq_h.bz.data(), array_size,
                                    cudaMemcpyHostToDevice, cu_streams.bz()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(ei, qq_h.ei.data(), array_size,
                                    cudaMemcpyHostToDevice, cu_streams.ei()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(ph, qq_h.ph.data(), array_size,
                                    cudaMemcpyHostToDevice, cu_streams.ph()));
    cudaDeviceSynchronize();
  }

  void copy_to_host(impl_host::Fields<Real> &qq_h, Streams &cu_streams) {
    cudaDeviceSynchronize();
    MISO_CUDA_CHECK(cudaMemcpyAsync(qq_h.ro.data(), ro, array_size,
                                    cudaMemcpyDeviceToHost, cu_streams.ro()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(qq_h.vx.data(), vx, array_size,
                                    cudaMemcpyDeviceToHost, cu_streams.vx()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(qq_h.vy.data(), vy, array_size,
                                    cudaMemcpyDeviceToHost, cu_streams.vy()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(qq_h.vz.data(), vz, array_size,
                                    cudaMemcpyDeviceToHost, cu_streams.vz()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(qq_h.bx.data(), bx, array_size,
                                    cudaMemcpyDeviceToHost, cu_streams.bx()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(qq_h.by.data(), by, array_size,
                                    cudaMemcpyDeviceToHost, cu_streams.by()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(qq_h.bz.data(), bz, array_size,
                                    cudaMemcpyDeviceToHost, cu_streams.bz()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(qq_h.ei.data(), ei, array_size,
                                    cudaMemcpyDeviceToHost, cu_streams.ei()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(qq_h.ph.data(), ph, array_size,
                                    cudaMemcpyDeviceToHost, cu_streams.ph()));
    cudaDeviceSynchronize();
  }

  void copy_from_device(const Fields<Real> &qq_d, Streams &cu_streams) {
    cudaDeviceSynchronize();
    MISO_CUDA_CHECK(cudaMemcpyAsync(ro, qq_d.ro, array_size,
                                    cudaMemcpyDeviceToDevice, cu_streams.ro()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(vx, qq_d.vx, array_size,
                                    cudaMemcpyDeviceToDevice, cu_streams.vx()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(vy, qq_d.vy, array_size,
                                    cudaMemcpyDeviceToDevice, cu_streams.vy()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(vz, qq_d.vz, array_size,
                                    cudaMemcpyDeviceToDevice, cu_streams.vz()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(bx, qq_d.bx, array_size,
                                    cudaMemcpyDeviceToDevice, cu_streams.bx()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(by, qq_d.by, array_size,
                                    cudaMemcpyDeviceToDevice, cu_streams.by()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(bz, qq_d.bz, array_size,
                                    cudaMemcpyDeviceToDevice, cu_streams.bz()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(ei, qq_d.ei, array_size,
                                    cudaMemcpyDeviceToDevice, cu_streams.ei()));
    MISO_CUDA_CHECK(cudaMemcpyAsync(ph, qq_d.ph, array_size,
                                    cudaMemcpyDeviceToDevice, cu_streams.ph()));
    cudaDeviceSynchronize();
  }
};

/// @brief Lightweight non-owning view of MHD buffers
template <typename Real> struct BuffersView {
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

  template <typename BuffersType>
  explicit BuffersView(const BuffersType &buffers)
      : recv_x_pos(buffers.recv_x_pos), recv_x_neg(buffers.recv_x_neg),
        recv_y_pos(buffers.recv_y_pos), recv_y_neg(buffers.recv_y_neg),
        recv_z_pos(buffers.recv_z_pos), recv_z_neg(buffers.recv_z_neg),
        send_x_pos(buffers.send_x_pos), send_x_neg(buffers.send_x_neg),
        send_y_pos(buffers.send_y_pos), send_y_neg(buffers.send_y_neg),
        send_z_pos(buffers.send_z_pos), send_z_neg(buffers.send_z_neg) {}
};

/// @brief MHD communication buffers on GPU
template <typename Real> struct Buffers {
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

  Buffers(const GridDevice<Real> &grid) {
    const auto buff_x_size =
        sizeof(Real) * grid.i_margin * grid.j_total * grid.k_total * n_vars;
    const auto buff_y_size =
        sizeof(Real) * grid.j_margin * grid.i_total * grid.k_total * n_vars;
    const auto buff_z_size =
        sizeof(Real) * grid.k_margin * grid.i_total * grid.j_total * n_vars;
    MISO_CUDA_CHECK(cudaMalloc(&recv_x_pos, buff_x_size));
    MISO_CUDA_CHECK(cudaMalloc(&recv_x_neg, buff_x_size));
    MISO_CUDA_CHECK(cudaMalloc(&recv_y_pos, buff_y_size));
    MISO_CUDA_CHECK(cudaMalloc(&recv_y_neg, buff_y_size));
    MISO_CUDA_CHECK(cudaMalloc(&recv_z_pos, buff_z_size));
    MISO_CUDA_CHECK(cudaMalloc(&recv_z_neg, buff_z_size));
    MISO_CUDA_CHECK(cudaMalloc(&send_x_pos, buff_x_size));
    MISO_CUDA_CHECK(cudaMalloc(&send_x_neg, buff_x_size));
    MISO_CUDA_CHECK(cudaMalloc(&send_y_pos, buff_y_size));
    MISO_CUDA_CHECK(cudaMalloc(&send_y_neg, buff_y_size));
    MISO_CUDA_CHECK(cudaMalloc(&send_z_pos, buff_z_size));
    MISO_CUDA_CHECK(cudaMalloc(&send_z_neg, buff_z_size));
  }

  ~Buffers() {
    const auto F = [](Real *&p) {
      if (p)
        MISO_CUDA_CHECK(cudaFree(p));
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

  // Shallow-const / shallow-copy
  BuffersView<Real> view() const noexcept { return BuffersView<Real>(*this); }

  // Prohibit copy and move operations
  Buffers(const Buffers &) = delete;
  Buffers &operator=(const Buffers &) = delete;
  Buffers(Buffers &&) = delete;
  Buffers &operator=(Buffers &&) = delete;
};

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
  const size_t buf_idx = ((i * grid.j_total + j) * grid.k_total + k) * n_vars;

  const Real *__restrict__ var[n_vars] = {qq_trgt.ro, qq_trgt.vx, qq_trgt.vy,
                                          qq_trgt.vz, qq_trgt.bx, qq_trgt.by,
                                          qq_trgt.bz, qq_trgt.ei, qq_trgt.ph};

#pragma unroll
  for (int m = 0; m < n_vars; ++m) {
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
  const size_t buf_idx = ((i * grid.j_margin + j) * grid.k_total + k) * n_vars;

  const Real *__restrict__ var[n_vars] = {qq_trgt.ro, qq_trgt.vx, qq_trgt.vy,
                                          qq_trgt.vz, qq_trgt.bx, qq_trgt.by,
                                          qq_trgt.bz, qq_trgt.ei, qq_trgt.ph};

#pragma unroll
  for (int m = 0; m < n_vars; ++m) {
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
  const size_t buf_idx = ((i * grid.j_total + j) * grid.k_margin + k) * n_vars;

  const Real *__restrict__ var[n_vars] = {qq_trgt.ro, qq_trgt.vx, qq_trgt.vy,
                                          qq_trgt.vz, qq_trgt.bx, qq_trgt.by,
                                          qq_trgt.bz, qq_trgt.ei, qq_trgt.ph};

#pragma unroll
  for (int m = 0; m < n_vars; ++m) {
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
  const size_t buf_idx = ((i * grid.j_total + j) * grid.k_total + k) * n_vars;

  Real *__restrict__ var[n_vars] = {qq_trgt.ro, qq_trgt.vx, qq_trgt.vy,
                                    qq_trgt.vz, qq_trgt.bx, qq_trgt.by,
                                    qq_trgt.bz, qq_trgt.ei, qq_trgt.ph};

#pragma unroll
  for (int m = 0; m < n_vars; ++m) {
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
  const size_t buf_idx = ((i * grid.j_margin + j) * grid.k_total + k) * n_vars;

  Real *__restrict__ var[n_vars] = {qq_trgt.ro, qq_trgt.vx, qq_trgt.vy,
                                    qq_trgt.vz, qq_trgt.bx, qq_trgt.by,
                                    qq_trgt.bz, qq_trgt.ei, qq_trgt.ph};

#pragma unroll
  for (int m = 0; m < n_vars; ++m) {
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
  const size_t buf_idx = ((i * grid.j_total + j) * grid.k_margin + k) * n_vars;

  Real *__restrict__ var[n_vars] = {qq_trgt.ro, qq_trgt.vx, qq_trgt.vy,
                                    qq_trgt.vz, qq_trgt.bx, qq_trgt.by,
                                    qq_trgt.bz, qq_trgt.ei, qq_trgt.ph};

#pragma unroll
  for (int m = 0; m < n_vars; ++m) {
    var[m][tgt_idx] = recv_buff[buf_idx + m];
  }
};

template <typename Real> struct HaloExchanger {
  Buffers<Real> buff;
  GridDevice<Real> &grid;
  mpi::Shape &mpi_shape;
  cuda::KernelShape3D &cu_shape;

  explicit HaloExchanger(GridDevice<Real> &grid, ExecContext &exec_ctx)
      : buff(grid), grid(grid), mpi_shape(exec_ctx.mpi_shape),
        cu_shape(exec_ctx.cu_shape) {}

  void apply(Fields<Real> &qq_trgt) {
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
                grid.i_margin * grid.j_total * grid.k_total * n_vars,
                mpi::data_type<Real>(), mpi_shape.x_procs_pos, 100,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(buff.recv_x_pos,
                grid.i_margin * grid.j_total * grid.k_total * n_vars,
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
                grid.i_total * grid.j_margin * grid.k_total * n_vars,
                mpi::data_type<Real>(), mpi_shape.y_procs_pos, 300,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(buff.recv_y_pos,
                grid.i_total * grid.j_margin * grid.k_total * n_vars,
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
                grid.i_total * grid.j_total * grid.k_margin * n_vars,
                mpi::data_type<Real>(), mpi_shape.z_procs_pos, 500,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(buff.recv_z_pos,
                grid.i_total * grid.j_total * grid.k_margin * n_vars,
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
                grid.i_margin * grid.j_total * grid.k_total * n_vars,
                mpi::data_type<Real>(), mpi_shape.x_procs_neg, 200,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(buff.recv_x_neg,
                grid.i_margin * grid.j_total * grid.k_total * n_vars,
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
                grid.i_total * grid.j_margin * grid.k_total * n_vars,
                mpi::data_type<Real>(), mpi_shape.y_procs_neg, 400,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(buff.recv_y_neg,
                grid.i_total * grid.j_margin * grid.k_total * n_vars,
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
                grid.i_total * grid.j_total * grid.k_margin * n_vars,
                mpi::data_type<Real>(), mpi_shape.z_procs_neg, 600,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(buff.recv_z_neg,
                grid.i_total * grid.j_total * grid.k_margin * n_vars,
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

}  // namespace impl_cuda
}  // namespace mhd
}  // namespace miso
