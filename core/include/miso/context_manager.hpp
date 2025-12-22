/// @brief Initializer/Finalizer of MISO
#pragma once

#include <mpi.h>

#include <miso/cuda_compat.hpp>

namespace miso {

/// @brief Initialize and finalize MPI environment.
struct MPIEnvironment {
  MPI_Comm comm = MPI_COMM_WORLD;
  int myrank = -1;
  int n_procs = -1;

  void setup() {
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &n_procs);
  }

  MPIEnvironment() {
    MPI_Init(nullptr, nullptr);
    setup();
  }

  MPIEnvironment(int &argc, char **&argv) {
    MPI_Init(&argc, &argv);
    setup();
  }

  ~MPIEnvironment() { MPI_Finalize(); }

  inline bool is_root() const noexcept { return myrank == 0; }
};

/// @brief Initialize and finalize CUDA environment.
#ifdef USE_CUDA
struct CUDAEnvironment {
  int device_cout;
  int device_id;

  CUDAEnvironment(MPIEnvironment &mpi_env) {
    cudaGetDeviceCount(&device_cout);
    device_id = mpi_env.myrank % device_cout;
    cudaSetDevice(device_id);
  }

  inline bool is_root() const noexcept { return device_id == 0; }
};
#endif

/// @brief Context manager for YAML, MPI, CUDA, and others.
/// @details This instance should be created at the beginning of main().
struct ContextManager {
  MPIEnvironment mpi_env;
#ifdef USE_CUDA
  CUDAEnvironment cuda_env;
#endif

  ContextManager()
#ifdef USE_CUDA
      : mpi_env(), cuda_env(mpi_env)
#else
      : mpi_env()
#endif
  {
  }

  ContextManager(int &argc, char **&argv)
#ifdef USE_CUDA
      : mpi_env(argc, argv), cuda_env(mpi_env)
#else
      : mpi_env(argc, argv)
#endif
  {
  }
};

}  // namespace miso
