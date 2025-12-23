#pragma once

#include <stdexcept>

#include <mpi.h>

#include <miso/cuda_compat.hpp>

namespace miso {

/// @brief Global environment for MPI
namespace mpi {

// global (multi-node) MPI info
namespace world {
inline bool is_initialized = false;
inline MPI_Comm comm = MPI_COMM_NULL;
inline int rank = -1;
inline int size = -1;
}  // namespace world

// local (per-node) MPI info for GPU mapping
namespace local {
inline MPI_Comm comm = MPI_COMM_NULL;
inline int rank = -1;
inline int size = -1;
}  // namespace local

/// @brief Initialize and finalize MPI environment.
struct Env {
  bool owns_mpi = false;

  void setup(int *argc, char ***argv) {
    int is_mpi_initialized = false;
    MPI_Initialized(&is_mpi_initialized);
    if (!is_mpi_initialized) {
      owns_mpi = true;
      MPI_Init(argc, argv);
    }
    world::comm = MPI_COMM_WORLD;
    MPI_Comm_rank(world::comm, &world::rank);
    MPI_Comm_size(world::comm, &world::size);
    MPI_Comm_split_type(world::comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local::comm);
    MPI_Comm_rank(local::comm, &local::rank);
    MPI_Comm_size(local::comm, &local::size);
    world::is_initialized = true;
  }

  Env() { setup(nullptr, nullptr); }
  Env(int &argc, char **&argv) { setup(&argc, &argv); }

  ~Env() noexcept {
    if (local::comm != MPI_COMM_NULL) {
      MPI_Comm_free(&local::comm);
      local::comm = MPI_COMM_NULL;
    }

    int is_mpi_finalized = false;
    MPI_Finalized(&is_mpi_finalized);
    if (owns_mpi && !is_mpi_finalized) {
      MPI_Finalize();
    }
  }
};

/// @brief Check if MPI is initialized.
inline bool is_initialized() { return world::is_initialized; }

/// @brief Get the global MPI communicator.
inline MPI_Comm comm() {
  if (!is_initialized()) {
    throw std::runtime_error(
        "mpi::Env must be instantiated at the beginning of the program.");
  }
  return world::comm;
}

/// @brief Get the rank of the current process.
inline int rank() {
  if (!is_initialized()) {
    throw std::runtime_error(
        "mpi::Env must be instantiated at the beginning of the program.");
  }
  return world::rank;
}

/// @brief Get the total number of processes.
inline int size() {
  if (!is_initialized()) {
    throw std::runtime_error(
        "mpi::Env must be instantiated at the beginning of the program.");
  }
  return world::size;
}

/// @brief Check if the current process is the root process.
inline bool is_root() { return rank() == 0; }

/// @brief get the local rank of the current process on the node
inline int local_rank() {
  if (!is_initialized()) {
    throw std::runtime_error(
        "mpi::Env must be instantiated at the beginning of the program.");
  }
  return local::rank;
}

/// @brief get the number of local processes on the node
inline int local_size() {
  if (!is_initialized()) {
    throw std::runtime_error(
        "mpi::Env must be instantiated at the beginning of the program.");
  }
  return local::size;
}

}  // namespace mpi

/// @brief Global environment for CUDA
#ifdef USE_CUDA
namespace cuda {

namespace internal {
inline bool is_initialized = false;
inline int device_count = 0;
inline int device_id = -1;
}  // namespace internal

/// @brief Check if CUDA is initialized.
inline bool is_initialized() { return internal::is_initialized; }

/// @brief Get the number of CUDA devices available.
inline int device_count() {
  if (!is_initialized()) {
    throw std::runtime_error(
        "CUDA must be initialized at the beginning of the program.");
  }
  return internal::device_count;
}

/// @brief Get the CUDA device ID assigned to the current process.
inline int device_id() {
  if (!is_initialized()) {
    throw std::runtime_error(
        "CUDA must be initialized at the beginning of the program.");
  }
  return internal::device_id;
}

/// @brief Initialize and finalize CUDA environment.
struct Env {
  Env() {
    if (!mpi::is_initialized()) {
      throw std::runtime_error(
          "MPI must be initialized before initializing CUDA.");
    }
    if (!internal::is_initialized) {
      internal::is_initialized = true;
    }
    cudaGetDeviceCount(&internal::device_count);
    if (internal::device_count == 0) {
      throw std::runtime_error("No CUDA-capable devices were detected.");
    }
    internal::device_id = mpi::local_rank() % internal::device_count;
    cudaSetDevice(internal::device_id);
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

  Env()
#ifdef USE_CUDA
      : mpi_env(), cuda_env()
#else
      : mpi_env()
#endif
  {
  }

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
