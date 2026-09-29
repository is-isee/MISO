#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <string>

#include <doctest/doctest.h>

#include <miso/core.hpp>
#include <miso/mhd_resistivity.hpp>

// Test of ResistiveSource::dt_limit() with MPI: the limit
// cfl_number / max(eta * sum(1/ds^2)) is the global one on every rank, and
// invalid eta (negative or not finite) is rejected on every rank.

namespace test_resistivity_dt {

using Real = double;
using namespace miso;

inline int rank() {
  int r;
  MPI_Comm_rank(mpi::comm(), &r);
  return r;
}

/// @brief Set eta = f(rank, i, j, k) in all cells and return it on the backend
template <typename Backend>
Array3D<Real, Backend> make_eta(const Grid<Real, backend::Host> &g,
                                const std::function<Real(int, int, int)> &f) {
  Array3D<Real, backend::Host> h(g.i_total, g.j_total, g.k_total);
  for (int i = 0; i < g.i_total; ++i) {
    for (int j = 0; j < g.j_total; ++j) {
      for (int k = 0; k < g.k_total; ++k) {
        h(i, j, k) = f(i, j, k);
      }
    }
  }
  Array3D<Real, Backend> d(g.i_total, g.j_total, g.k_total);
  d.copy_from(h);
  return d;
}

template <typename Backend> void check_dt_limit() {
  Config config(std::string(CONFIG_DIR) + "config_source.yaml");
  mpi::Shape mpi_shape(config);
  Grid<Real, backend::Host> g(config, mpi_shape);
  Grid<Real, Backend> grid(g);
  // uniform grid: 8 x 6 x 4 cells in [0, 1]^3
  const Real sum_dsi2 = 8.0 * 8.0 + 6.0 * 6.0 + 4.0 * 4.0;
  const Real cfl = config["mhd"]["resistivity"]["cfl_number"].as<Real>();
  REQUIRE(cfl == 0.5);  // default value

  SUBCASE("The maximum over all ranks is used on every rank") {
    // eta = 0.01 on rank 0, 0.02 on rank 1, with a small spatial variation
    auto eta = make_eta<Backend>(g, [&](int i, int, int) {
      return 0.01 * (rank() + 1) * (1.0 + 0.1 * g.x[i]);
    });
    Real eta_max = 0;
    for (int i = g.i_margin; i < g.i_total - g.i_margin; ++i) {
      eta_max = std::max(eta_max, 0.01 * (rank() + 1) * (1.0 + 0.1 * g.x[i]));
    }
    Real eta_max_g;
    MPI_Allreduce(&eta_max, &eta_max_g, 1, MPI_DOUBLE, MPI_MAX, mpi::comm());
    if (rank() == 0) {
      // the global maximum is on the other rank
      REQUIRE(eta_max_g > eta_max * 1.5);
    }
    const mhd::ResistiveSource<Real, Backend> src(config, grid, eta);
    CHECK(src.dt_limit() ==
          doctest::Approx(cfl / (eta_max_g * sum_dsi2)).epsilon(1e-14));
  }

  SUBCASE("No limit for eta = 0") {
    auto eta = make_eta<Backend>(g, [](int, int, int) { return 0.0; });
    const mhd::ResistiveSource<Real, Backend> src(config, grid, eta);
    CHECK(src.dt_limit() >= 1.e10);
  }

  SUBCASE("update_dt_limit follows a change of eta") {
    auto eta = make_eta<Backend>(g, [](int, int, int) { return 0.01; });
    const mhd::ResistiveSource<Real, Backend> src(config, grid, eta);
    CHECK(src.dt_limit() ==
          doctest::Approx(cfl / (0.01 * sum_dsi2)).epsilon(1e-14));
    eta.copy_from(make_eta<backend::Host>(g, [](int, int, int) { return 0.04; }));
    src.update_dt_limit();
    CHECK(src.dt_limit() ==
          doctest::Approx(cfl / (0.04 * sum_dsi2)).epsilon(1e-14));
  }

  SUBCASE("Invalid eta on one rank is rejected on every rank") {
    const Real bad[3] = {-1.e-3, std::numeric_limits<Real>::quiet_NaN(),
                         std::numeric_limits<Real>::infinity()};
    for (const Real b : bad) {
      CAPTURE(b);
      // a single bad cell (a ghost cell) on rank 1 only
      auto eta = make_eta<Backend>(g, [&](int i, int j, int k) {
        return (rank() == 1 && i == 0 && j == 0 && k == 0) ? b : 0.01;
      });
      const mhd::ResistiveSource<Real, Backend> src(config, grid, eta);
      CHECK_THROWS_AS(src.dt_limit(), std::runtime_error);
    }
  }
}

}  // namespace test_resistivity_dt
