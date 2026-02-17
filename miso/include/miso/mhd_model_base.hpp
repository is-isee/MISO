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
#include "utility.hpp"

namespace miso {

template <class Derived, class Backend> class ModelBase {
public:
  using Real = miso::Real;
  using EOS = eos::IdealEOS<Real>;
  using MHD = mhd::MHD<Real, EOS, Backend>;

  Config &config;
  mpi::Shape mpi_shape;
  Time<Real> time;
  Grid<Real, backend::Host> grid;

  mhd::ExecContext<Backend> exec_ctx;
  EOS eos;
  MHD mhd;

  // デフォルト実装で使うものだけ保持
  mhd::NoSource<Real> default_src;

  explicit ModelBase(Config &cfg)
      : config(cfg), mpi_shape(cfg), time(cfg), grid(cfg, mpi_shape),
        exec_ctx(mpi_shape, grid), eos(cfg), mhd(cfg, grid, exec_ctx, eos),
        default_src() {}

  // ----- デフォルトフック（派生が同名メンバ関数を持てば隠蔽される） -----
  auto &source() { return default_src; }
  Real compute_dt() { return mhd.cfl(); }
  void after_initial_condition() {}

  void save_grid_metadata() { grid.save(config); }  // 最小例はこれでよい

  // ---- 共通I/O ----
  void save_metadata() {
    MPI_Barrier(mpi::comm());
    config.save();
    derived().save_grid_metadata();  // ←派生で上書き可
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
    auto &d = derived();

    // ic/bc は「メンバがある前提」にしたいなら static_assert を足す
    d.mhd.apply_initial_condition(d.ic, d.bc);

    d.after_initial_condition();

    if (config["base"]["continue"].as<bool>() &&
        fs::exists(time.time_save_dir + "n_output.txt")) {
      load_state();
    }

    save_metadata();
    save_if_needed();

    while (time.time < time.tend) {
      const auto dt = d.compute_dt();      // ←派生で差し替え可
      d.mhd.update(dt, d.bc, d.source());  // ←source() を差し替え可
      d.time.update(dt);
      save_if_needed();
    }
  }

protected:
  Derived &derived() { return static_cast<Derived &>(*this); }
};

}  // namespace miso
