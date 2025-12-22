#pragma once

#include <mpi.h>

#include <miso/cuda_compat.hpp>

namespace miso {

/// @brief Global environment for MPI
namespace mpi {
inline bool is_initialized = false;
inline MPI_Comm comm;
inline int myrank = -1;
inline int n_procs = -1;
inline bool is_root() noexcept { return myrank == 0; }

/// @brief Initialize and finalize MPI environment.
struct Env {
  void setup() {
    mpi::comm = MPI_COMM_WORLD;
    MPI_Comm_rank(mpi::comm, &mpi::myrank);
    MPI_Comm_size(mpi::comm, &mpi::n_procs);
  }

  Env(int &argc, char **&argv) {
    if (!mpi::is_initialized) {
      MPI_Init(&argc, &argv);
      mpi::is_initialized = true;
    }
    setup();
  }

  ~Env() { MPI_Finalize(); }
};
}  // namespace mpi

/// @brief Global environment for CUDA
#ifdef USE_CUDA
namespace cuda {
inline bool is_initialized = false;
inline int device_count = 0;
inline int device_id = -1;
inline bool is_root() noexcept { return device_id == 0; }

/// @brief Initialize and finalize CUDA environment.
struct Env {
  Env() {
    if (!mpi::is_initialized) {
      throw std::runtime_error(
          "MPI must be initialized before initializing CUDA.");
    }
    if (!cuda::is_initialized) {
      cuda::is_initialized = true;
    }
    cudaGetDeviceCount(&cuda::device_count);
    cuda::device_id = mpi::myrank % cuda::device_count;
    cudaSetDevice(cuda::device_id);
  }
};
}  // namespace cuda
#endif

/// @brief Environment manager for MPI and CUDA environments.
/// @details This instance should be created at the beginning of main().
struct Env {
  mpi::Env mpi_env;
#ifdef USE_CUDA
  cuda::Env cuda_env;
#endif

  Env(int &argc, char **&argv)
#ifdef USE_CUDA
      : mpi_env(argc, argv), cuda_env()
#else
      : mpi_env(argc, argv)
#endif
  {
  }
};

}  // namespace miso
