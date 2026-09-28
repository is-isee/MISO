#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_mhd_resistivity_common.hpp"

using namespace test_resistivity;

namespace {

const Orientation orients[3] = {orient_xyz, orient_yzx, orient_zxy};

/// @brief Value of `a` at problem index (iu, iv, iw) on the oriented grid
Real at_uvw(const Array3D_h &a, const Grid_h &g, const Orientation &o,
            std::array<int, 3> iuvw) {
  const std::array<int, 3> margin{g.i_margin, g.j_margin, g.k_margin};
  std::array<int, 3> idx{0, 0, 0};
  for (int a_ = 0; a_ < 3; ++a_) {
    idx[o.axis[a_]] = iuvw[a_] + margin[o.axis[a_]];
  }
  return a(idx[0], idx[1], idx[2]);
}

struct Solved {
  Grid_h grid;
  Tendency t;
  Solved(std::array<int, 3> n, const Orientation &o, const VecFunc &b,
         const ScalarFunc &eta_f)
      : grid(make_grid(n, o)), t(solve(grid, o, b, eta_f)) {}

  static Tendency solve(const Grid_h &g, const Orientation &o, const VecFunc &b,
                        const ScalarFunc &eta_f) {
    Fields_h qq(g);
    Array3D_h eta(g.i_total, g.j_total, g.k_total);
    set_problem(g, o, b, eta_f, qq, eta);
    return tendency_host(g, eta, qq);
  }
};

}  // namespace

TEST_CASE("Resistivity: same result on every axis and plane" *
          doctest::test_suite("resistivity")) {
  // 1D (x, y, z), 2D (xy, yz, zx) and 3D (cyclic permutations)
  const std::array<int, 3> sizes[3] = {{16, 1, 1}, {16, 12, 1}, {12, 10, 8}};
  for (const auto &n : sizes) {
    CAPTURE(n[0]);
    CAPTURE(n[1]);
    CAPTURE(n[2]);
    Solved s0(n, orients[0], b_smooth, eta_smooth);
    Real scale = 0;
    for (int m = 0; m < 3; ++m) {
      for (int idx = 0; idx < s0.t.db(m).size(); ++idx) {
        scale = std::max(scale, std::abs(s0.t.db(m)[idx]));
      }
    }
    REQUIRE(scale > 0);

    for (int oi = 1; oi < 3; ++oi) {
      const Orientation &o = orients[oi];
      Solved s1(n, o, b_smooth, eta_smooth);
      for (int iu = 0; iu < n[0]; ++iu) {
        for (int iv = 0; iv < n[1]; ++iv) {
          for (int iw = 0; iw < n[2]; ++iw) {
            const std::array<int, 3> id{iu, iv, iw};
            for (int a = 0; a < 3; ++a) {
              const Real v0 =
                  at_uvw(s0.t.db(orients[0].axis[a]), s0.grid, orients[0], id);
              const Real v1 = at_uvw(s1.t.db(o.axis[a]), s1.grid, o, id);
              CHECK(std::abs(v1 - v0) <= 1.e-12 * scale);
            }
            const Real e0 = at_uvw(s0.t.de, s0.grid, orients[0], id);
            const Real e1 = at_uvw(s1.t.de, s1.grid, o, id);
            CHECK(std::abs(e1 - e0) <= 1.e-12 * scale);
          }
        }
      }
    }
  }
}

TEST_CASE("Resistivity: second-order convergence" *
          doctest::test_suite("resistivity")) {
  // error ratio for doubled resolution should be about 4
  const std::array<int, 3> base[3] = {{16, 1, 1}, {16, 16, 1}, {8, 8, 8}};
  for (const auto &n : base) {
    for (const auto &o : orients) {
      CAPTURE(n[0]);
      CAPTURE(n[1]);
      CAPTURE(n[2]);
      CAPTURE(o.axis[0]);
      std::array<int, 3> n2, n4;
      for (int a = 0; a < 3; ++a) {
        n2[a] = n[a] > 1 ? 2 * n[a] : 1;
        n4[a] = n[a] > 1 ? 4 * n[a] : 1;
      }
      const Real e1 = max_error(n, o);
      const Real e2 = max_error(n2, o);
      const Real e4 = max_error(n4, o);
      MESSAGE("errors: ", e1, " ", e2, " ", e4);
      CHECK(e1 / e2 > 3.5);
      CHECK(e2 / e4 > 3.8);
    }
  }
}

TEST_CASE("Resistivity: grid-scale (checkerboard) modes are damped" *
          doctest::test_suite("resistivity")) {
  // (-1)^(i+j+...) is represented by sin(N pi u) sin(N pi v) ... at cell
  // centers. With uniform eta, the compact normal derivative gives the exact
  // damping rate 4 eta / du^2 per transverse direction. A scheme that applies
  // centered differences twice would give zero (odd-even decoupling).
  const Real eta0 = 0.01;
  const ScalarFunc eta_u = [=](Real, Real, Real) { return eta0; };
  const int nu = 8, nv = 6, nw = 4;
  const Real du = 1.0 / nu, dv = 1.0 / nv, dw = 1.0 / nw;
  auto cb = [](int n, Real s) { return std::sin(n * M_PI * s); };

  for (const auto &o : orients) {
    CAPTURE(o.axis[0]);
    // 2D (u, v) plane: w component and u component
    {
      const VecFunc b = [&](Real u, Real v, Real) {
        const Real c = cb(nu, u) * cb(nv, v);
        return std::array<Real, 3>{c, 0.0, c};
      };
      Solved s({nu, nv, 1}, o, b, eta_u);
      const Real rate_w = -4 * eta0 * (1 / (du * du) + 1 / (dv * dv));
      const Real rate_u = -4 * eta0 / (dv * dv);
      for (int iu = 0; iu < nu; ++iu) {
        for (int iv = 0; iv < nv; ++iv) {
          const Real c = ((iu + iv) % 2 == 0) ? 1.0 : -1.0;
          const std::array<int, 3> id{iu, iv, 0};
          CHECK(at_uvw(s.t.db(o.axis[2]), s.grid, o, id) ==
                doctest::Approx(rate_w * c).epsilon(1.e-10));
          CHECK(at_uvw(s.t.db(o.axis[0]), s.grid, o, id) ==
                doctest::Approx(rate_u * c).epsilon(1.e-10));
          CHECK(std::abs(at_uvw(s.t.db(o.axis[1]), s.grid, o, id)) <
                1.e-10 * std::abs(rate_w));
        }
      }
    }
    // 3D: u component
    {
      const VecFunc b = [&](Real u, Real v, Real w) {
        return std::array<Real, 3>{cb(nu, u) * cb(nv, v) * cb(nw, w), 0.0, 0.0};
      };
      Solved s({nu, nv, nw}, o, b, eta_u);
      const Real rate_u = -4 * eta0 * (1 / (dv * dv) + 1 / (dw * dw));
      for (int iu = 0; iu < nu; ++iu) {
        for (int iv = 0; iv < nv; ++iv) {
          for (int iw = 0; iw < nw; ++iw) {
            const Real c = ((iu + iv + iw) % 2 == 0) ? 1.0 : -1.0;
            const std::array<int, 3> id{iu, iv, iw};
            CHECK(at_uvw(s.t.db(o.axis[0]), s.grid, o, id) ==
                  doctest::Approx(rate_u * c).epsilon(1.e-10));
          }
        }
      }
    }
  }
}

TEST_CASE("Resistivity: energy conservation and dissipation" *
          doctest::test_suite("resistivity")) {
  for (const auto &o : orients) {
    CAPTURE(o.axis[0]);
    const std::array<int, 3> n{12, 10, 8};
    Grid_h g = make_grid(n, o);
    Fields_h qq(g);
    Array3D_h eta(g.i_total, g.j_total, g.k_total);
    set_problem(g, o, b_smooth, eta_smooth, qq, eta);
    const Tendency t = tendency_host(g, eta, qq);

    Real de_sum = 0, de_abs = 0, dmag_sum = 0;
    for (int i = g.i_margin; i < g.i_total - g.i_margin; ++i) {
      for (int j = g.j_margin; j < g.j_total - g.j_margin; ++j) {
        for (int k = g.k_margin; k < g.k_total - g.k_margin; ++k) {
          de_sum += t.de(i, j, k);
          de_abs += std::abs(t.de(i, j, k));
          dmag_sum += pii4<Real> * (qq.bx(i, j, k) * t.dbx(i, j, k) +
                                    qq.by(i, j, k) * t.dby(i, j, k) +
                                    qq.bz(i, j, k) * t.dbz(i, j, k));
        }
      }
    }
    REQUIRE(de_abs > 0);
    // total energy is conserved (flux form)
    CHECK(std::abs(de_sum) < 1.e-12 * de_abs);
    // magnetic energy decreases (Joule heating is positive in total)
    CHECK(dmag_sum < 0);
  }
}
