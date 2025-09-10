#pragma once

#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <yaml-cpp/yaml.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

template <typename Real> struct MPIManager {
  MPI_Comm cart_comm = MPI_COMM_NULL;
  int myrank = -1;
  int n_procs = -1;
  static constexpr int ndims = 3;
  int coord[ndims];
  int x_procs, y_procs, z_procs;
  int x_procs_pos, y_procs_pos, z_procs_pos;
  int x_procs_neg, y_procs_neg, z_procs_neg;
  int n_procs_digits;

  MPIManager() {
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  }

  ~MPIManager() {
    if (cart_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&cart_comm);
    }
    MPI_Finalize();
  }

  constexpr bool is_root() const noexcept { return myrank == 0; }

  void init_parameters(const YAML::Node &yaml_obj) {
    n_procs_digits = yaml_obj["mpi"]["n_procs_digits"].template as<int>();
    x_procs = yaml_obj["mpi"]["x_procs"].template as<int>();
    y_procs = yaml_obj["mpi"]["y_procs"].template as<int>();
    z_procs = yaml_obj["mpi"]["z_procs"].template as<int>();
    if (x_procs * y_procs * z_procs != n_procs) {
      if (myrank == 0) {
        std::cerr << "####################################################"
                  << std::endl;
        std::cerr << "Error: # of mpi procs != x_procs * y_procs * z_procs"
                  << std::endl;
        std::cerr << "mpi procs = " << n_procs << std::endl;
        std::cerr << "x_procs = " << x_procs << std::endl;
        std::cerr << "y_procs = " << y_procs << std::endl;
        std::cerr << "z_procs = " << z_procs << std::endl;
        std::cerr << "####################################################"
                  << std::endl;
      }
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      std::exit(EXIT_FAILURE);
    }
  }

  void set_cart_comm(const YAML::Node &yaml_obj) {
    int dims[ndims] = {x_procs, y_procs, z_procs};
    int periods[ndims];
    if (n_procs == 1) {
      periods[0] = 0;
      periods[1] = 0;
      periods[2] = 0;
    } else {
      periods[0] =
          yaml_obj["boundary_condition"]["periodic"]["x"].template as<bool>() ? 1
                                                                              : 0;
      periods[1] =
          yaml_obj["boundary_condition"]["periodic"]["y"].template as<bool>() ? 1
                                                                              : 0;
      periods[2] =
          yaml_obj["boundary_condition"]["periodic"]["z"].template as<bool>() ? 1
                                                                              : 0;
    }

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &cart_comm);
    MPI_Comm_rank(cart_comm, &myrank);
    MPI_Comm_size(cart_comm, &n_procs);

    MPI_Cart_coords(cart_comm, myrank, ndims, coord);
    int merr;
    merr = MPI_Cart_shift(cart_comm, 0, 1, &x_procs_neg, &x_procs_pos);
    merr = MPI_Cart_shift(cart_comm, 1, 1, &y_procs_neg, &y_procs_pos);
    merr = MPI_Cart_shift(cart_comm, 2, 1, &z_procs_neg, &z_procs_pos);
  }

  void setup_mpi(YAML::Node &yaml_obj) {
    init_parameters(yaml_obj);
    set_cart_comm(yaml_obj);

#ifdef USE_CUDA
    // TODO: この部分はcuda_manager.cuhに移したいが、設定の順序でうまく行っていない
    // TODO: 複数ノードを使う場合は、ここを修正する必要
    int device_count;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(myrank % device_count);
#endif
  }
};
