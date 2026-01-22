#pragma once

#include <miso/grid.hpp>
#include <miso/mhd_core.hpp>
#include <miso/backend.hpp>
#ifdef __CUDACC__
#include <miso/cuda_util.cuh>
#endif  // __CUDACC__

namespace miso {
namespace mhd {

/// @brief Lightweight non-owning view of MHD buffers.
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

/// @brief MHD communication buffers.
template <typename Real, typename Backend = backend::Host> struct Buffers;

/// @brief MHD communication buffers on GPU.
template <typename Real> struct Buffers<Real, backend::CUDA> {
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

  Buffers(const Grid<Real, backend::CUDA> &grid) {
    const auto buff_x_size =
        sizeof(Real) * grid.i_margin * grid.j_total * grid.k_total * n_fields;
    const auto buff_y_size =
        sizeof(Real) * grid.j_margin * grid.i_total * grid.k_total * n_fields;
    const auto buff_z_size =
        sizeof(Real) * grid.k_margin * grid.i_total * grid.j_total * n_fields;
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

}  // namespace mhd
}  // namespace miso
