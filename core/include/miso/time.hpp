#pragma once
#include <cassert>
#include <filesystem>

#include <mpi.h>

#include <miso/config.hpp>
#include <miso/mpi_types.hpp>
#include <miso/utility.hpp>

namespace miso {

/// @brief Class to manage time-related parameters and operations
/// @tparam Real
template <typename Real> struct Time {
  /// @brief current time
  Real time;
  /// @brief end time
  Real tend;
  /// @brief output time interval
  Real dt_output;
  /// @brief time step size
  Real dt;
  /// @brief time step number
  int n_step;
  /// @brief time step number for output
  int n_output;
  /// @brief number of digits for output file naming
  int n_output_digits;
  /// @brief directory for saving time-related files
  std::string time_save_dir;

  /// @brief Initialize time parameters
  void initialize() {
    dt = 0;
    time = 0;
    n_step = 0;
    n_output = 0;
  }

  /// @brief Default constructor
  Time(const Config &config)
      : tend(config["time"]["tend"].template as<Real>()),
        dt_output(config["time"]["dt_output"].template as<Real>()),
        n_output_digits(config["time"]["n_output_digits"].template as<int>()) {
    assert(tend > 0);
    assert(dt_output > 0);

    initialize();

    time_save_dir = config.save_dir +
                    config["time"]["time_save_dir"].template as<std::string>();
    util::create_directories(time_save_dir);
  }

  /// @brief update time parameters
  void update() {
    time += dt;
    n_step++;
  };

  /// @brief Save time parameters to file
  /// @param config
  void save(const Config &config) const {
    if (mpi::is_root()) {
      std::ostringstream fname;
      fname << time_save_dir << "/time." << util::zfill(n_output, n_output_digits)
            << ".txt";
      std::ofstream ofs(fname.str());
      assert(ofs.is_open());
      ofs << time << "\n";
      ofs << n_output << "\n";
      ofs << n_step << "\n";

      std::ofstream ofs_step(time_save_dir + "/n_output.txt");
      assert(ofs_step.is_open());
      ofs_step << n_output << "\n";
    }
  }

  /// @brief Load time parameters from file
  /// @param config
  void load(const Config &config) {
    if (mpi::is_root()) {
      std::ifstream ifs_step(time_save_dir + "/n_output.txt");
      ifs_step >> n_output;

      std::ostringstream fname;
      fname << time_save_dir << "/time." << util::zfill(n_output, n_output_digits)
            << ".txt";
      std::ifstream ifs(fname.str());
      if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open time file: " + fname.str());
      }

      ifs >> time;
      ifs >> n_output;
      ifs >> n_step;
    }

    MPI_Bcast(&time, 1, mpi_type<Real>(), 0, mpi::comm());
    MPI_Bcast(&n_output, 1, MPI_INT, 0, mpi::comm());
    MPI_Bcast(&n_step, 1, MPI_INT, 0, mpi::comm());
  }
};

}  // namespace miso
