#pragma once

#include "config.hpp"
#include "env.hpp"
#include "eos.hpp"
#include "execution.hpp"
#include "grid.hpp"
#include "mhd.hpp"
#include "mpi_util.hpp"
#include "time.hpp"
#include "types.hpp"

namespace miso {
namespace mhd {

/// @brief Base class of MHD models using CRTP.
/// @details The derived class must implement the following members:
/// - eos: equation of state
/// - ic: initial condition
/// - bc: boundary condition
/// - src: source term (optional; default is no source)
template <class Derived, class Real, class Backend> class ModelBase {
public:
  Config &config;
  mpi::Shape mpi_shape;
  Time<Real> time;
  Grid<Real, backend::Host> grid;

  ExecContext<Backend> exec_ctx;
  MHD<Real, Backend> mhd;

  explicit ModelBase(Config &cfg)
      : config(cfg), mpi_shape(cfg), time(cfg), grid(cfg, mpi_shape),
        exec_ctx(mpi_shape, grid), mhd(cfg, grid, exec_ctx) {}

  /// @brief Default CFL condition (override if needed)
  Real compute_dt() { return mhd.cfl(derived().eos); }

  void save_metadata() {
    MPI_Barrier(mpi::comm());
    config.save();
    grid.save(config);
    exec_ctx.mpi_shape.save();
  }

  void save_state() {
    time.save();
    mhd.save(time);
  }

  void load_state() {
    time.load();
    mhd.load(time);
  }

  void save_if_needed() {
    if (time.time >= time.dt_output * time.n_output) {
      save_state();
      time.log();
      time.n_output++;
    }
  }

  void run() {
    // Ensure to call methods in the derived class
    auto &d = derived();

    // Assert that d has eos, ic, bc, and src as member variables.
    static_assert(std::is_member_object_pointer_v<decltype(&Derived::eos)>,
                  "Derived class must have a member variable named 'eos' for "
                  "equation of state.");
    static_assert(std::is_member_object_pointer_v<decltype(&Derived::ic)>,
                  "Derived class must have a member variable named 'ic' for "
                  "initial condition.");
    static_assert(std::is_member_object_pointer_v<decltype(&Derived::bc)>,
                  "Derived class must have a member variable named 'bc' for "
                  "boundary condition.");
    static_assert(std::is_member_object_pointer_v<decltype(&Derived::src)>,
                  "Derived class must have a member variable named 'src' for "
                  "source term.");

    mhd.apply_initial_condition(d.ic, d.bc);
    if (config["base"]["continue"].as<bool>() &&
        fs::exists(time.time_save_dir + "n_output.txt")) {
      load_state();
    }

    save_metadata();
    save_if_needed();

    while (time.time < time.tend) {
      const auto dt = d.compute_dt();
      mhd.update(dt, d.eos, d.bc, d.src);
      time.update(dt);
      save_if_needed();
    }
  }

protected:
  Derived &derived() { return static_cast<Derived &>(*this); }
};

/// @brief Empty boundary condition class (e.g., periodic in all directions).
/// @details Periodic boundary condition is applied by MPI communication.
/// Be sure to set "periodic" in domain field of config.yaml.
struct EmptyBoundaryCondition {
  void apply(mhd::FieldsView<Real>, GridView<const Real>) const {}
};

/// @brief Empty source term class (without source terms).
/// @details Volumetric heat / force terms are expected.
template <typename Real> struct EmptySourceTerm {
  /// External force: x-direction
  __host__ __device__ inline Real vx(FieldsView<const Real>, int, int,
                                     int) const noexcept {
    return 0.0;
  }

  /// External force: y-direction
  __host__ __device__ inline Real vy(FieldsView<const Real>, int, int,
                                     int) const noexcept {
    return 0.0;
  }

  /// External force: z-direction
  __host__ __device__ inline Real vz(FieldsView<const Real>, int, int,
                                     int) const noexcept {
    return 0.0;
  }

  /// External heating
  __host__ __device__ inline Real ei(FieldsView<const Real>, int, int,
                                     int) const noexcept {
    return 0.0;
  }
};

}  // namespace mhd
}  // namespace miso
