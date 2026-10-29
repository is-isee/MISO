#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
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

void fill(Array3D<Real> &arr, Real value) {
  std::fill(arr.data(), arr.data() + arr.size(), value);
}

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

  // Env must be created once, so the cases are not doctest SUBCASEs.
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

  {
    // Blending: with I = S + c for every ray, J - S = c and F = 0, so
    // Q_J = 4 pi alpha c, Q_F = 0 and Q_rad = (1 - w) Q_J.
    const Atmosphere atm{10.0, -0.2, 0.0, 0.05};
    solve(atm, grid, mpi_shape, rt, heating);
    const Real c = 0.5;
    for (int m = 0; m < rt.ang_quad.num_rays; ++m) {
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          for (int k = 0; k < grid.k_total; ++k) {
            rt.rint(m, i, j, k) = atm.source(grid.z[k]) + c;
          }
        }
      }
    }
    heating.compute(rt, grid, mpi_shape);
    const Real q_j = 4 * pi<Real> * atm.alpha * c;
    for (int i = rt.ib0 + grid.is; i <= rt.ib1; ++i) {
      for (int j = rt.jb0 + grid.js; j <= rt.jb1; ++j) {
        for (int k = rt.kb0 + grid.ks; k <= rt.kb1; ++k) {
          const Real tau = atm.alpha * (z_top_global - grid.z[k]);
          const Real w = std::min(
              std::max(std::log(tau / 1e-2) / std::log(2e-1 / 1e-2), 0.0), 1.0);
          const Real expected = (tau < 1e-4) ? 0.0 : (1 - w) * q_j;
          CHECK(heating.qrad(i, j, k) ==
                doctest::Approx(expected).epsilon(1e-9).scale(q_j));
        }
      }
    }
  }

  {
    // S and alpha from the table in input_data
    const fs::path path = fs::path(INPUT_DATA_DIR) / "eos_table_roen_blend.tbl";
    eos::TableEOS<Real> eos(path);
    const Real ro_photo = std::exp(-15.377656523015137);
    const Real ei_photo = std::exp(27.981135057266535);
    Array3D<Real> ro(grid.i_total, grid.j_total, grid.k_total);
    Array3D<Real> ei(grid.i_total, grid.j_total, grid.k_total);
    fill(ro, ro_photo);
    fill(ei, ei_photo);
    rt::set_source_and_absorption(rt, grid, ro, ei, eos);
    const Real te = eos.temperature(ro_photo, ei_photo);
    CHECK(rt.src_func(1, 1, 1) == doctest::Approx(rt::stefan_boltzmann<Real> *
                                                  std::pow(te, 4) / pi<Real>));
    CHECK(rt.abs_coeff(1, 1, 1) ==
          doctest::Approx(ro_photo * eos.opacity(ro_photo, ei_photo)));

    // The opacity is inf at rho = 1.1e-11 g/cm^3, T = 2.1e6 K.
    ro(1, 1, 1) = 1.0944997993195436e-11;
    ei(1, 1, 1) = 460706556163877.0;
    CHECK_THROWS_AS(rt::set_source_and_absorption(rt, grid, ro, ei, eos),
                    std::runtime_error);
  }

  {
    // Vertical 2D domain in x-z (j_size = 1): the same solution as in 3D.
    Config config_xz(std::string(CONFIG_DIR) + "config_rt_heating_xz.yaml");
    mpi::Shape shape_xz(config_xz);
    Grid<Real> grid_xz(config_xz, shape_xz);
    REQUIRE(grid_xz.j_size == 1);
    REQUIRE(grid_xz.j_total == 1);
    rt::RT<Real> rt_xz(grid_xz, 24);
    rt::RadiativeHeating<Real> heating_xz(grid_xz);
    const Atmosphere atm{10.0, -0.2, 2.0e-3, 0.05};
    solve(atm, grid_xz, shape_xz, rt_xz, heating_xz);
    for (int i = rt_xz.ib0 + grid_xz.is; i <= rt_xz.ib1; ++i) {
      for (int k = rt_xz.kb0 + grid_xz.ks; k <= rt_xz.kb1; ++k) {
        const Real z = grid_xz.z[k];
        CHECK(heating_xz.flux_z(i, 0, k) ==
              doctest::Approx(atm.flux_z(z)).epsilon(0.01));
        CHECK(std::abs(heating_xz.flux_y(i, 0, k)) <
              1e-12 * 4 * pi<Real> * atm.a);
        if (heating_xz.tau(i, 0, k) >= heating_xz.tau_gate) {
          CHECK(heating_xz.qrad(i, 0, k) ==
                doctest::Approx(atm.heating()).epsilon(0.01));
        }
      }
    }
  }

  {
    // Horizontal flux across two ranks in x (non-periodic). With
    //   I(mu) = S + mu_x g x + mu_z h z  (S and alpha constant),
    // J = S, F_x = 4 pi g x / 3, F_z = 4 pi h z / 3 and Q_rad = -4 pi (g + h) / 3
    // where tau >= tau_thick (all cells but the topmost one).
    Config config_x(std::string(CONFIG_DIR) + "config_rt_heating_x.yaml");
    mpi::Shape shape_x(config_x);
    Grid<Real> grid_x(config_x, shape_x);
    REQUIRE(shape_x.x_procs == 2);
    rt::RT<Real> rt_x(grid_x, 24);
    rt::RadiativeHeating<Real> heating_x(grid_x);

    const Real s0 = 10.0, g = 0.3, h = -0.2;
    fill(rt_x.src_func, s0);
    fill(rt_x.abs_coeff, 1.0);
    const auto &q = rt_x.ang_quad;
    for (int m = 0; m < q.num_rays; ++m) {
      for (int i = 0; i < grid_x.i_total; ++i) {
        for (int j = 0; j < grid_x.j_total; ++j) {
          for (int k = 0; k < grid_x.k_total; ++k) {
            rt_x.rint(m, i, j, k) =
                s0 + q.mu_x[m] * g * grid_x.x[i] + q.mu_z[m] * h * grid_x.z[k];
          }
        }
      }
    }
    heating_x.compute(rt_x, grid_x, shape_x);
    const Real expected = -4 * pi<Real> * (g + h) / 3;
    for (int i = rt_x.ib0 + grid_x.is; i <= rt_x.ib1; ++i) {
      for (int j = rt_x.jb0 + grid_x.js; j <= rt_x.jb1; ++j) {
        for (int k = rt_x.kb0 + grid_x.ks; k <= rt_x.kb1; ++k) {
          CHECK(heating_x.flux_x(i, j, k) ==
                doctest::Approx(4 * pi<Real> * g * grid_x.x[i] / 3));
          if (k < rt_x.kb1 || shape_x.z_procs_pos != MPI_PROC_NULL) {
            CHECK(heating_x.qrad(i, j, k) == doctest::Approx(expected));
          }
        }
      }
    }
  }
}
