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

template <class Derived, class Backend> class ModelBase {
public:
  using Real = miso::Real;
  using EOS = eos::IdealEOS<Real>;  // Default EOS
  using MHD = mhd::MHD<Real, EOS, Backend>;

  Config &config;
  mpi::Shape mpi_shape;
  Time<Real> time;
  Grid<Real, backend::Host> grid;

  mhd::ExecContext<Backend> exec_ctx;
  EOS eos;
  MHD mhd;

  mhd::NoSource<Real> default_src;

  explicit ModelBase(Config &cfg)
      : config(cfg), mpi_shape(cfg), time(cfg), grid(cfg, mpi_shape),
        exec_ctx(mpi_shape, grid), eos(cfg), mhd(cfg, grid, exec_ctx, eos),
        default_src() {}

  /// @brief Default source term (no source; override if needed)
  auto &source() { return default_src; }

  /// @brief Default CFL condition (override if needed)
  Real compute_dt() { return mhd.cfl(); }

  /// @brief Default update method (override if needed)
  void update() {
    // Ensure to call methods in the derived class (e.g., applying_initial_condition)
    auto &d = derived();

    const auto dt = d.compute_dt();
    d.mhd.update(dt, d.bc, d.source());
    d.time.update(dt);
  }

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
    if (time.time < time.dt_output * time.n_output)
      return;
    save_state();
    time.log();
    time.n_output++;
  }

  void run() {
    // Ensure to call methods in the derived class (e.g., applying_initial_condition)
    auto &d = derived();

    static_assert(std::is_member_function_pointer_v<
                      decltype(&Derived::apply_initial_condition)>,
                  "Derived class must have apply_initial_condition method");
    d.mhd.apply_initial_condition(d.ic, d.bc);

    if (config["base"]["continue"].as<bool>() &&
        fs::exists(time.time_save_dir + "n_output.txt")) {
      load_state();
    }

    save_metadata();
    save_if_needed();

    while (time.time < time.tend) {
      d.update();
      save_if_needed();
    }
  }

protected:
  Derived &derived() { return static_cast<Derived &>(*this); }
};

}  // namespace miso
