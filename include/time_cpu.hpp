#pragma once
#include "config.hpp"
#include "mpi_types.hpp"
#include "utility.hpp"
#include <cassert>
#include <mpi.h>

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

  /// @brief Initialize time parameters
  void initialize() {
    dt = 0;
    time = 0;
    n_step = 0;
    n_output = 0;
  }

  /// @brief Default constructor
  /// @param yaml_obj YAML node read in Config class
  Time(const YAML::Node &yaml_obj)
      : tend(yaml_obj["time"]["tend"].template as<Real>()),
        dt_output(yaml_obj["time"]["dt_output"].template as<Real>()),
        n_output_digits(yaml_obj["time"]["n_output_digits"].template as<int>()) {
    assert(tend > 0);
    assert(dt_output > 0);

    initialize();
  }

  /// @brief Constructor with parameters
  /// @param tend_ end time
  /// @param dt_output_ output time interval
  /// @param n_output_digits_ number of digits for output file naming
  Time(Real tend_, Real dt_output_, int n_output_digits_)
      : tend(tend_), dt_output(dt_output_), n_output_digits(n_output_digits_) {
    assert(tend > 0);
    assert(dt_output > 0);

    initialize();
  }

  /// @brief update time parameters
  void update() {
    time += dt;
    n_step++;
  };

  /// @brief Save time parameters to file
  /// @param config
  void save(const Config &config) const {
    if (config.mpi.myrank == 0) {
      std::ostringstream fname;
      fname << config.time_save_dir << "/time."
            << util::zfill(this->n_output, this->n_output_digits) << ".txt";
      std::ofstream ofs(fname.str());
      assert(ofs.is_open());
      ofs << time << "\n";
      ofs << this->n_output << "\n";
      ofs << this->n_step << "\n";

      std::ofstream ofs_step(config.time_save_dir + "/n_output.txt");
      assert(ofs_step.is_open());
      ofs_step << this->n_output << "\n";
    }
  }

  /// @brief Load time parameters from file
  /// @param config
  void load(const Config &config) {
    if (config.mpi.myrank == 0) {
      std::ifstream ifs_step(config.time_save_dir + "/n_output.txt");
      ifs_step >> this->n_output;

      std::ostringstream fname;
      fname << config.time_save_dir << "/time."
            << util::zfill(this->n_output, this->n_output_digits) << ".txt";
      std::ifstream ifs(fname.str());
      if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open time file: " + fname.str());
      }

      ifs >> this->time;
      ifs >> this->n_output;
      ifs >> this->n_step;
    }

    MPI_Bcast(&this->time, 1, mpi_type<Real>(), 0, config.mpi.cart_comm);
    MPI_Bcast(&this->n_output, 1, MPI_INT, 0, config.mpi.cart_comm);
    MPI_Bcast(&this->n_step, 1, MPI_INT, 0, config.mpi.cart_comm);
  }
};