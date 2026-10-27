#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>

#include <miso/config.hpp>
#include <miso/env.hpp>
#include <miso/grid.hpp>
#include <miso/mpi_util.hpp>
#include <miso/rt.hpp>

using namespace miso;

namespace {

using Real = double;

// S = a + b z + c z^2 with a constant alpha. The intensity
//   I(mu) = S - mu S' / alpha + mu^2 S'' / alpha^2
// solves dI/ds = alpha (S - I) along a ray with direction cosine mu (to z).
// Then, with sum w mu_z^2 = 1/3 and sum w mu_z^3 = 0,
//   J = S + S'' / (3 alpha^2),  F_z = -4 pi S' / (3 alpha),
//   Q_J = Q_F = 4 pi S'' / (3 alpha).
struct Atmosphere {
  Real a, b, c, alpha;
  Real source(Real z) const { return a + b * z + c * z * z; }
  Real intensity(Real z, Real mu) const {
    const Real ds = b + 2 * c * z;
    return source(z) - mu * ds / alpha + mu * mu * 2 * c / (alpha * alpha);
  }
  Real flux_z(Real z) const {
    return -4 * pi<Real> * (b + 2 * c * z) / (3 * alpha);
  }
  Real heating() const { return 4 * pi<Real> * 2 * c / (3 * alpha); }
};

// Exact intensity for the rays coming in through the z boundaries
struct ExactBoundary {
  Atmosphere atm;
  void operator()(rt::RT<Real> &rt, const Grid<Real> &grid,
                  const mpi::Shape &mpi_shape) const {
    const auto &q = rt.ang_quad;
    for (int m = 0; m < q.num_rays; ++m) {
      int k = -1;
      if (q.mu_z[m] > 0 && mpi_shape.z_procs_neg == MPI_PROC_NULL) {
        k = rt.kb0;
      } else if (q.mu_z[m] < 0 && mpi_shape.z_procs_pos == MPI_PROC_NULL) {
        k = rt.kb1;
      }
      if (k < 0) {
        continue;
      }
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          rt.rint(m, i, j, k) = atm.intensity(grid.z[k], q.mu_z[m]);
        }
      }
    }
  }
};

void solve(const Atmosphere &atm, const Grid<Real> &grid,
           const mpi::Shape &mpi_shape, rt::RT<Real> &rt,
           rt::RadiativeHeating<Real> &heating) {
  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      for (int k = 0; k < grid.k_total; ++k) {
        rt.src_func(i, j, k) = atm.source(grid.z[k]);
        rt.abs_coeff(i, j, k) = atm.alpha;
      }
    }
  }
  rt.solve(grid, mpi_shape, Real(1e-12), 200, ExactBoundary{atm});
  heating.compute(rt, grid, mpi_shape);
}

}  // namespace

TEST_CASE("Q_rad in an atmosphere with a known solution" *
          doctest::test_suite("rt_heating")) {
  Env env;
  Config config(std::string(CONFIG_DIR) + "config_rt_heating.yaml");
  mpi::Shape mpi_shape(config);
  Grid<Real> grid(config, mpi_shape);
  REQUIRE(mpi_shape.z_procs == 2);

  rt::RT<Real> rt(grid, 24);
  rt::RadiativeHeating<Real> heating(grid);
  const Real z_top = grid.z[rt.kb1];  // topmost RT point of this rank
  Real z_top_global = 0;
  MPI_Allreduce(&z_top, &z_top_global, 1, mpi::data_type<Real>(), MPI_MAX,
                mpi_shape.cart_comm);

  // Env must be created once, so the two cases are not doctest SUBCASEs.
  {
    // Linear source function: no heating, constant flux.
    // Optically thin at the top (tau < 0.2 in the top 4 cells).
    const Atmosphere atm{10.0, -0.2, 0.0, 0.05};
    solve(atm, grid, mpi_shape, rt, heating);
    // Size of each term of Q_J = 4 pi alpha (J - S)
    const Real q_scale = 4 * pi<Real> * atm.alpha * atm.a;
    for (int i = rt.ib0 + grid.is; i <= rt.ib1; ++i) {
      for (int j = rt.jb0 + grid.js; j <= rt.jb1; ++j) {
        for (int k = rt.kb0 + grid.ks; k <= rt.kb1; ++k) {
          const Real z = grid.z[k];
          CHECK(heating.tau(i, j, k) ==
                doctest::Approx(atm.alpha * (z_top_global - z)));
          CHECK(heating.mean_intensity(i, j, k) ==
                doctest::Approx(atm.source(z)));
          CHECK(heating.flux_z(i, j, k) == doctest::Approx(atm.flux_z(z)));
          CHECK(std::abs(heating.flux_x(i, j, k)) < 1e-12 * 4 * pi<Real> * atm.a);
          CHECK(std::abs(heating.qrad(i, j, k)) < 1e-8 * q_scale);
        }
      }
    }
  }

  {
    // Quadratic source function: Q_J and Q_F agree.
    const Atmosphere atm{10.0, -0.2, 2.0e-3, 0.05};
    solve(atm, grid, mpi_shape, rt, heating);
    for (int i = rt.ib0 + grid.is; i <= rt.ib1; ++i) {
      for (int j = rt.jb0 + grid.js; j <= rt.jb1; ++j) {
        for (int k = rt.kb0 + grid.ks; k <= rt.kb1; ++k) {
          if (heating.tau(i, j, k) < heating.tau_gate) {
            CHECK(heating.qrad(i, j, k) == 0);
          } else {
            CHECK(heating.qrad(i, j, k) ==
                  doctest::Approx(atm.heating()).epsilon(0.01));
          }
        }
      }
    }
  }
}
