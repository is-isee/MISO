#pragma once

#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

#include <doctest/doctest.h>

#include <miso/core.hpp>
#include <miso/mhd.hpp>
#include <miso/mhd_resistivity.hpp>

// Test of ResistiveSource::dt_limit() with MPI.
// - dt_limit() returns the limit of each rank and follows a change of eta.
// - ModelBase::update uses the minimum over all ranks.
// - Invalid eta (negative or not finite) is rejected: by the constructor on
//   the rank, and by ModelBase::update on every rank.

namespace test_resistivity_dt {

using Real = double;
using namespace miso;

inline int rank() {
  int r;
  MPI_Comm_rank(mpi::comm(), &r);
  return r;
}

inline std::string config_path() {
  return std::string(CONFIG_DIR) + "config_source.yaml";
}

// uniform grid: 8 x 6 x 4 cells in [0, 1]^3 (config_source.yaml)
constexpr Real sum_dsi2 = 8.0 * 8.0 + 6.0 * 6.0 + 4.0 * 4.0;

/// @brief Set eta = f(i, j, k) in all cells of `eta`
template <typename Backend>
void set_eta(Array3D<Real, Backend> &eta, const Grid<Real, backend::Host> &g,
             const std::function<Real(int, int, int)> &f) {
  Array3D<Real, backend::Host> h(g.i_total, g.j_total, g.k_total);
  for (int i = 0; i < g.i_total; ++i) {
    for (int j = 0; j < g.j_total; ++j) {
      for (int k = 0; k < g.k_total; ++k) {
        h(i, j, k) = f(i, j, k);
      }
    }
  }
  eta.copy_from(h);
}

inline Real uniform_eta(Real value, int, int, int) { return value; }

/// @brief Model with ResistiveSource on a uniform state (B = 0)
template <typename Backend>
struct Model : public mhd::ModelBase<Model<Backend>, Real, Backend> {
  struct Uniform {
    eos::IdealEOS<Real> &eos;
    void apply(mhd::FieldsView<Real> qq, GridView<const Real>) const {
      for (int n = 0; n < qq.size(); ++n) {
        qq.ro[n] = 1.0;
        qq.vx[n] = qq.vy[n] = qq.vz[n] = 0.0;
        qq.bx[n] = qq.by[n] = qq.bz[n] = 0.0;
        qq.ei[n] = 1.0 / (eos.gm - 1.0);
        qq.ph[n] = 0.0;
      }
    }
  };
  eos::IdealEOS<Real> eos;
  Uniform ic;
  mhd::EmptyBoundaryCondition<Real> bc;
  Array3D<Real, Backend> eta;
  mhd::ResistiveSource<Real, Backend> src;

  Model(Config &config, Real eta_rank0, Real eta_rank1)
      : mhd::ModelBase<Model<Backend>, Real, Backend>(config), eos(config),
        ic{eos}, bc(),
        eta(this->grid.i_total, this->grid.j_total, this->grid.k_total),
        src(config, this->mhd.grid,
            (set_eta(eta, this->grid,
                     [&](int, int, int) {
                       return rank() == 0 ? eta_rank0 : eta_rank1;
                     }),
             eta)) {}
};

template <typename Backend> void check_dt_limit() {
  Config config(config_path());
  mpi::Shape mpi_shape(config);
  Grid<Real, backend::Host> g(config, mpi_shape);
  Grid<Real, Backend> grid(g);
  const Real cfl = config["mhd"]["resistivity"]["cfl_number"].as<Real>();
  REQUIRE(cfl == 0.5);  // default value
  Array3D<Real, Backend> eta(g.i_total, g.j_total, g.k_total);

  SUBCASE("dt_limit() is the limit of each rank") {
    // eta = 0.01 on rank 0, 0.02 on rank 1, with a small spatial variation
    const auto f = [&](int i, int, int) {
      return 0.01 * (rank() + 1) * (1.0 + 0.1 * g.x[i]);
    };
    set_eta(eta, g, f);
    Real eta_max = 0;
    for (int i = g.i_margin; i < g.i_total - g.i_margin; ++i) {
      eta_max = std::max(eta_max, f(i, 0, 0));
    }
    const mhd::ResistiveSource<Real, Backend> src(config, grid, eta);
    CHECK(src.dt_limit() ==
          doctest::Approx(cfl / (eta_max * sum_dsi2)).epsilon(1e-14));
  }

  SUBCASE("No limit for eta = 0") {
    set_eta(eta, g, [](int, int, int) { return 0.0; });
    const mhd::ResistiveSource<Real, Backend> src(config, grid, eta);
    CHECK(src.dt_limit() >= 1.e10);
  }

  SUBCASE("dt_limit() follows a change of eta (time-dependent eta)") {
    set_eta(eta, g, [](int, int, int) { return 0.01; });
    const mhd::ResistiveSource<Real, Backend> src(config, grid, eta);
    CHECK(src.dt_limit() ==
          doctest::Approx(cfl / (0.01 * sum_dsi2)).epsilon(1e-14));
    set_eta(eta, g, [](int, int, int) { return 0.04; });
    CHECK(src.dt_limit() ==
          doctest::Approx(cfl / (0.04 * sum_dsi2)).epsilon(1e-14));
  }

  SUBCASE("Invalid eta is rejected") {
    const Real bad[3] = {-1.e-3, std::numeric_limits<Real>::quiet_NaN(),
                         std::numeric_limits<Real>::infinity()};
    for (const Real b : bad) {
      CAPTURE(b);
      // a single bad cell (a ghost cell) on rank 1 only
      const auto f = [&](int i, int j, int k) {
        return (rank() == 1 && i == 0 && j == 0 && k == 0) ? b : 0.01;
      };
      // by the constructor, on the rank with the bad cell
      set_eta(eta, g, f);
      const auto construct = [&]() {
        const mhd::ResistiveSource<Real, Backend> src(config, grid, eta);
      };
      if (rank() == 1) {
        CHECK_THROWS_AS(construct(), std::runtime_error);
      } else {
        CHECK_NOTHROW(construct());
      }
      // by dt_limit() (negative on that rank) after eta is changed
      set_eta(eta, g, [](int, int, int) { return 0.01; });
      const mhd::ResistiveSource<Real, Backend> src(config, grid, eta);
      set_eta(eta, g, f);
      if (rank() == 1) {
        CHECK(src.dt_limit() < 0);
      } else {
        CHECK(src.dt_limit() > 0);
      }
    }
  }
}

/// @brief ModelBase::update with ResistiveSource
template <typename Backend> void check_model_update() {
  Config config(config_path());
  const Real cfl = config["mhd"]["resistivity"]["cfl_number"].as<Real>();

  SUBCASE("The minimum over all ranks is used") {
    // eta = 0.1 on rank 0, 0.2 on rank 1: the diffusive limit of rank 1
    // (0.5 / (0.2 * 116) = 0.0216) is smaller than the CFL limit (0.053),
    // and that of rank 0 (0.0431) is between them.
    Model<Backend> model(config, 0.1, 0.2);
    model.mhd.apply_initial_condition(model.ic, model.bc);
    model.update();
    CHECK(model.time.time ==
          doctest::Approx(cfl / (0.2 * sum_dsi2)).epsilon(1e-14));
  }

  SUBCASE("Invalid eta on one rank throws on every rank") {
    Model<Backend> model(config, 0.01, 0.01);
    model.mhd.apply_initial_condition(model.ic, model.bc);
    set_eta(model.eta, model.grid,
            [](int, int, int) { return rank() == 1 ? -1.0 : 0.01; });
    CHECK_THROWS_AS(model.update(), std::runtime_error);
  }
}

}  // namespace test_resistivity_dt
