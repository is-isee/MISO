/// @brief Initializer/Finalizer of MISO
#pragma once

#include <miso/runtime_manager.hpp>

namespace miso {

/// @brief Context manager for YAML, MPI, CUDA, and others.
/// @details This instance should be created at the beginning of main().
struct ContextManager {
  MPIRuntime mpi_rt;
#ifdef USE_CUDA
  CUDARuntime cuda_env;
#endif

  ContextManager()
#ifdef USE_CUDA
      : mpi_rt(), cuda_env(mpi_rt)
#else
      : mpi_rt()
#endif
  {
  }

  ContextManager(int &argc, char **&argv)
#ifdef USE_CUDA
      : mpi_rt(argc, argv), cuda_env(mpi_rt)
#else
      : mpi_rt(argc, argv)
#endif
  {
  }
};

}  // namespace miso
