#pragma once

#include <array>
#include <cmath>
#include <functional>

#include <doctest/doctest.h>

#include <miso/array3d.hpp>
#include <miso/constants.hpp>
#include <miso/grid.hpp>
#include <miso/mhd_fields.hpp>
#include <miso/mhd_resistivity.hpp>

// Unit tests of the explicit resistive terms (mhd::ResistiveSource).
//
// A test problem is defined in coordinates (u, v, w) in [0, 1)^3 with periodic
// boundaries and mapped onto the (x, y, z) axes by a cyclic permutation, so
// that the same problem can be solved on any axis or plane (x, y, z, xy, yz,
// zx) and the results can be compared with each other.

namespace test_resistivity {

using Real = double;
using namespace miso;
using Grid_h = Grid<Real, backend::Host>;
using Fields_h = mhd::Fields<Real, backend::Host>;
using Array3D_h = Array3D<Real, backend::Host>;

constexpr Real pi2 = 2.0 * M_PI;

/// @brief Vector (u, v, w components) as a function of (u, v, w)
using VecFunc = std::function<std::array<Real, 3>(Real, Real, Real)>;
using ScalarFunc = std::function<Real(Real, Real, Real)>;

/// @brief Mapping from (u, v, w) to (x, y, z): axis[a] is the (x, y, z) axis
/// of the a-th problem axis.
struct Orientation {
  std::array<int, 3> axis;
};
inline constexpr Orientation orient_xyz{{0, 1, 2}};
inline constexpr Orientation orient_yzx{{1, 2, 0}};
inline constexpr Orientation orient_zxy{{2, 0, 1}};

/// @brief Tendency of B (per component) and total energy
struct Tendency {
  Array3D_h dbx, dby, dbz, de;
  explicit Tendency(const Grid_h &g)
      : dbx(g.i_total, g.j_total, g.k_total),
        dby(g.i_total, g.j_total, g.k_total),
        dbz(g.i_total, g.j_total, g.k_total),
        de(g.i_total, g.j_total, g.k_total) {}
  const Array3D_h &db(int m) const { return m == 0 ? dbx : (m == 1 ? dby : dbz); }
};

/// @brief Problem sizes along (u, v, w) mapped to the (x, y, z) grid
inline Grid_h make_grid(std::array<int, 3> n_uvw, const Orientation &o,
                        int margin = 2) {
  std::array<int, 3> n_xyz{};
  for (int a = 0; a < 3; ++a) {
    n_xyz[o.axis[a]] = n_uvw[a];
  }
  return Grid_h(n_xyz[0], n_xyz[1], n_xyz[2], margin, 0.0, 1.0, 0.0, 1.0, 0.0,
                1.0);
}

inline int wrap(int i, int size, int margin) {
  if (size == 1) {
    return 0;
  }
  return margin + (((i - margin) % size) + size) % size;
}

/// @brief Copy interior values to ghost cells periodically (edges/corners too)
inline void fill_periodic(Array3D_h &a, const Grid_h &g) {
  for (int i = 0; i < g.i_total; ++i) {
    for (int j = 0; j < g.j_total; ++j) {
      for (int k = 0; k < g.k_total; ++k) {
        a(i, j, k) =
            a(wrap(i, g.i_size, g.i_margin), wrap(j, g.j_size, g.j_margin),
              wrap(k, g.k_size, g.k_margin));
      }
    }
  }
}

/// @brief Coordinates (u, v, w) of the cell (i, j, k)
inline std::array<Real, 3> uvw_of(const Grid_h &g, const Orientation &o, int i,
                                  int j, int k) {
  const std::array<Real, 3> xyz{g.x[i], g.y[j], g.z[k]};
  return {xyz[o.axis[0]], xyz[o.axis[1]], xyz[o.axis[2]]};
}

/// @brief Set B (given in (u, v, w) components) and eta on the grid.
inline void set_problem(const Grid_h &g, const Orientation &o, const VecFunc &b,
                        const ScalarFunc &eta_f, Fields_h &qq, Array3D_h &eta) {
  Array3D_h *bxyz[3] = {&qq.bx, &qq.by, &qq.bz};
  for (int i = g.i_margin; i < g.i_total - g.i_margin; ++i) {
    for (int j = g.j_margin; j < g.j_total - g.j_margin; ++j) {
      for (int k = g.k_margin; k < g.k_total - g.k_margin; ++k) {
        const auto p = uvw_of(g, o, i, j, k);
        const auto bb = b(p[0], p[1], p[2]);
        for (int a = 0; a < 3; ++a) {
          (*bxyz[o.axis[a]])(i, j, k) = bb[a];
        }
        eta(i, j, k) = eta_f(p[0], p[1], p[2]);
        qq.ro(i, j, k) = 1.0;
        qq.vx(i, j, k) = qq.vy(i, j, k) = qq.vz(i, j, k) = 0.0;
        qq.ei(i, j, k) = 1.0;
        qq.ph(i, j, k) = 0.0;
      }
    }
  }
  for (auto *a : {&qq.ro, &qq.vx, &qq.vy, &qq.vz, &qq.bx, &qq.by, &qq.bz, &qq.ei,
                  &qq.ph, &eta}) {
    fill_periodic(*a, g);
  }
}

/// @brief Evaluate the resistive source terms (ResistiveSource) on the host.
inline Tendency tendency_host(const Grid_h &g, const Array3D_h &eta,
                              const Fields_h &qq) {
  const mhd::ResistiveSource<Real, backend::Host> src(g, eta, Real(0.5));
  const auto q = qq.const_view();
  Tendency t(g);
  for (int n = 0; n < qq.ro.size(); ++n) {
    t.dbx[n] = t.dby[n] = t.dbz[n] = t.de[n] = 0;
  }
  for (int i = g.i_margin; i < g.i_total - g.i_margin; ++i) {
    for (int j = g.j_margin; j < g.j_total - g.j_margin; ++j) {
      for (int k = g.k_margin; k < g.k_total - g.k_margin; ++k) {
        t.dbx(i, j, k) = src.bx(q, i, j, k);
        t.dby(i, j, k) = src.by(q, i, j, k);
        t.dbz(i, j, k) = src.bz(q, i, j, k);
        t.de(i, j, k) = src.ei(q, i, j, k);
      }
    }
  }
  const int ic = g.i_total / 2, jc = g.j_total / 2, kc = g.k_total / 2;
  CHECK(src.vx(q, ic, jc, kc) == 0);
  CHECK(src.vy(q, ic, jc, kc) == 0);
  CHECK(src.vz(q, ic, jc, kc) == 0);
  return t;
}

// ---------------------------------------------------------------------------
// Test problems (smooth, periodic in [0, 1)^3)

inline std::array<Real, 3> b_smooth(Real u, Real v, Real w) {
  return {std::sin(pi2 * v) + 0.3 * std::cos(pi2 * (u + w)),
          0.5 * std::cos(pi2 * u) + std::sin(pi2 * (u + v)) +
              0.2 * std::sin(pi2 * w),
          std::cos(pi2 * u) * std::sin(pi2 * v) + 0.4 * std::cos(pi2 * w)};
}

inline Real eta_smooth(Real u, Real v, Real w) {
  return 0.01 + 0.005 * std::sin(pi2 * u) * std::cos(pi2 * v) +
         0.002 * std::cos(pi2 * w);
}

/// @brief Reference tendency by nested centered differences of the
/// continuous fluxes: dB_i/dt = d_j G_ij, dE/dt = d_j ((1/4pi) B_i G_ij),
/// G_ij = eta (d_j B_i - d_i B_j). Returns {dB_u, dB_v, dB_w, dE}.
/// @param active Directions in which the problem varies
inline std::array<Real, 4> reference_tendency(const VecFunc &b,
                                              const ScalarFunc &eta_f,
                                              std::array<Real, 3> p,
                                              std::array<bool, 3> active) {
  const Real h = 1.e-3;
  auto shifted = [](std::array<Real, 3> q, int d, Real s) {
    q[d] += s;
    return q;
  };
  auto bcomp = [&](std::array<Real, 3> q, int i) {
    return b(q[0], q[1], q[2])[i];
  };
  auto deriv = [&](std::array<Real, 3> q, int i, int j) {  // d_j B_i
    if (!active[j]) {
      return Real(0);
    }
    return (bcomp(shifted(q, j, h), i) - bcomp(shifted(q, j, -h), i)) / (2 * h);
  };
  auto gg = [&](std::array<Real, 3> q, int i, int j) {
    return eta_f(q[0], q[1], q[2]) * (deriv(q, i, j) - deriv(q, j, i));
  };
  auto fe = [&](std::array<Real, 3> q, int j) {
    Real s = 0;
    for (int i = 0; i < 3; ++i) {
      s += bcomp(q, i) * gg(q, i, j);
    }
    return pii4<Real> * s;
  };
  std::array<Real, 4> r{0, 0, 0, 0};
  for (int j = 0; j < 3; ++j) {
    if (!active[j]) {
      continue;
    }
    for (int i = 0; i < 3; ++i) {
      r[i] +=
          (gg(shifted(p, j, h), i, j) - gg(shifted(p, j, -h), i, j)) / (2 * h);
    }
    r[3] += (fe(shifted(p, j, h), j) - fe(shifted(p, j, -h), j)) / (2 * h);
  }
  return r;
}

/// @brief Maximum error of the tendency against the reference
inline Real max_error(std::array<int, 3> n_uvw, const Orientation &o) {
  Grid_h g = make_grid(n_uvw, o);
  Fields_h qq(g);
  Array3D_h eta(g.i_total, g.j_total, g.k_total);
  set_problem(g, o, b_smooth, eta_smooth, qq, eta);
  const Tendency t = tendency_host(g, eta, qq);
  const std::array<bool, 3> active{n_uvw[0] > 1, n_uvw[1] > 1, n_uvw[2] > 1};

  Real err = 0;
  for (int i = g.i_margin; i < g.i_total - g.i_margin; ++i) {
    for (int j = g.j_margin; j < g.j_total - g.j_margin; ++j) {
      for (int k = g.k_margin; k < g.k_total - g.k_margin; ++k) {
        const auto p = uvw_of(g, o, i, j, k);
        const auto r = reference_tendency(b_smooth, eta_smooth, p, active);
        for (int a = 0; a < 3; ++a) {
          err = std::max(err, std::abs(t.db(o.axis[a])(i, j, k) - r[a]));
        }
        err = std::max(err, std::abs(t.de(i, j, k) - r[3]));
      }
    }
  }
  return err;
}

}  // namespace test_resistivity
