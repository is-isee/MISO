#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>

#include <mpi.h>
#include <yaml-cpp/yaml.h>

#include <miso/config.hpp>
#include <miso/env.hpp>

namespace miso {
namespace mpi {

inline void check_mpi_error(int merr, const char *msg, MPI_Comm comm) {
  if (merr != MPI_SUCCESS) {
    std::cerr << "Error in " << msg << std::endl;
    MPI_Abort(comm, merr);
  }
}

struct Manager {
  MPI_Comm cart_comm = MPI_COMM_NULL;
  int myrank = -1;
  int n_procs = -1;
  static constexpr int ndims = 3;
  int coord[ndims];
  int x_procs, y_procs, z_procs;
  int x_procs_pos, y_procs_pos, z_procs_pos;
  int x_procs_neg, y_procs_neg, z_procs_neg;
  std::string mpi_save_dir;

  Manager(const Config &config) {
    mpi_save_dir = config.save_dir +
                   config["mpi"]["mpi_save_dir"].template as<std::string>();
    util::create_directories(mpi_save_dir);

    x_procs = config["mpi"]["x_procs"].template as<int>();
    y_procs = config["mpi"]["y_procs"].template as<int>();
    z_procs = config["mpi"]["z_procs"].template as<int>();

    int dims[ndims] = {x_procs, y_procs, z_procs};
    int periods[ndims];
    periods[0] = config["domain"]["periodic"]["x"].template as<bool>() ? 1 : 0;
    periods[1] = config["domain"]["periodic"]["y"].template as<bool>() ? 1 : 0;
    periods[2] = config["domain"]["periodic"]["z"].template as<bool>() ? 1 : 0;

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &cart_comm);
    MPI_Comm_rank(cart_comm, &myrank);
    MPI_Comm_size(cart_comm, &n_procs);
    if (x_procs * y_procs * z_procs != n_procs) {
      if (mpi::is_root()) {
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

    MPI_Cart_coords(cart_comm, myrank, ndims, coord);
    int merr;
    merr = MPI_Cart_shift(cart_comm, 0, 1, &x_procs_neg, &x_procs_pos);
    check_mpi_error(merr, "MPI_Cart_shift x", cart_comm);
    merr = MPI_Cart_shift(cart_comm, 1, 1, &y_procs_neg, &y_procs_pos);
    check_mpi_error(merr, "MPI_Cart_shift y", cart_comm);
    merr = MPI_Cart_shift(cart_comm, 2, 1, &z_procs_neg, &z_procs_pos);
    check_mpi_error(merr, "MPI_Cart_shift z", cart_comm);
  }

  ~Manager() {
    if (cart_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&cart_comm);
    }
  }

  void save() const {
    int all_coords[ndims * n_procs];
    MPI_Gather(coord, ndims, MPI_INT, all_coords, ndims, MPI_INT, 0, cart_comm);

    if (mpi::is_root()) {
      std::ofstream ofs(mpi_save_dir + "/coords.csv");
      ofs << "rank,x,y,z\n";
      for (int rank = 0; rank < n_procs; ++rank) {
        ofs << rank << "," << all_coords[rank * 3 + 0] << ","
            << all_coords[rank * 3 + 1] << "," << all_coords[rank * 3 + 2]
            << "\n";
      }
    }
  }
};

}  // namespace mpi
}  // namespace miso
