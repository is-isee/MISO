#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>

#include "config.hpp"
#include "env.hpp"

// clang-format off
/// @brief Macro to check MPI errors
#define MISO_MPI_CHECK(merr) \
  do { miso::mpi::check_error((merr), __FILE__, __LINE__); } while (0)
// clang-format on

namespace miso {
namespace mpi {

inline void check_error(int merr, const char *file, int line, bool abort = true) {
  if (merr != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int len = 0;
    int rc = MPI_Error_string(merr, errstr, &len);
    if (rc == MPI_SUCCESS) {
      // Ensure null-termination
      if (len < 0)
        len = 0;
      if (len >= MPI_MAX_ERROR_STRING)
        len = MPI_MAX_ERROR_STRING - 1;
      errstr[len] = '\0';
      std::fprintf(stderr, "MPI Error: %s %s %d\n", errstr, file, line);
    } else {
      std::fprintf(stderr, "MPI Error: %d %s %d\n", merr, file, line);
    }
    std::fflush(stderr);
    if (abort)
      MPI_Abort(mpi::comm(), EXIT_FAILURE);
  }
}

/// @brief MPI Datatype corresponding to Real type
template <typename Real> MPI_Datatype data_type();
template <> inline MPI_Datatype data_type<float>() { return MPI_FLOAT; }
template <> inline MPI_Datatype data_type<double>() { return MPI_DOUBLE; }

/// @brief Define Cartesian shape in MPI process topology
struct Shape {
  MPI_Comm cart_comm = MPI_COMM_NULL;
  int myrank = -1;
  int n_procs = -1;
  static constexpr int ndims = 3;
  int coord[ndims];
  int x_procs, y_procs, z_procs;
  int x_procs_pos, y_procs_pos, z_procs_pos;
  int x_procs_neg, y_procs_neg, z_procs_neg;
  std::string mpi_save_dir;

  Shape(const Config &config) {
    mpi_save_dir =
        config.save_dir + config["mpi"]["mpi_save_dir"].as<std::string>();
    util::create_directories(mpi_save_dir);

    x_procs = config["mpi"]["x_procs"].as<int>();
    y_procs = config["mpi"]["y_procs"].as<int>();
    z_procs = config["mpi"]["z_procs"].as<int>();

    int dims[ndims] = {x_procs, y_procs, z_procs};
    int periods[ndims];
    periods[0] = config["domain"]["periodic"]["x"].as<bool>() ? 1 : 0;
    periods[1] = config["domain"]["periodic"]["y"].as<bool>() ? 1 : 0;
    periods[2] = config["domain"]["periodic"]["z"].as<bool>() ? 1 : 0;

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
    MISO_MPI_CHECK(MPI_Cart_shift(cart_comm, 0, 1, &x_procs_neg, &x_procs_pos));
    MISO_MPI_CHECK(MPI_Cart_shift(cart_comm, 1, 1, &y_procs_neg, &y_procs_pos));
    MISO_MPI_CHECK(MPI_Cart_shift(cart_comm, 2, 1, &z_procs_neg, &z_procs_pos));
  }

  ~Shape() {
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
