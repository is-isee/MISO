#pragma once

namespace miso {

/// @brief Memory policy for host memory.
struct HostSpace {};

/// @brief Memory policy for CUDA device memory.
struct CUDASpace {};

/// @brief Execution policy for host.
struct HostBackend {
  using memory_space = HostSpace;
};

/// @brief Execution policy for CUDA.
struct CUDABackend {
  using memory_space = CUDASpace;
};

}  // namespace miso
