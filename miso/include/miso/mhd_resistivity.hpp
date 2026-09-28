#pragma once

#include "array3d.hpp"
#include "constants.hpp"
#include "execution.hpp"
#include "grid.hpp"
#include "mhd_fields.hpp"
#include "mpi_util.hpp"
#include "utility.hpp"

namespace miso {
namespace mhd {

namespace impl_resistivity {

/// @brief Offset of a cell index by `s` in direction `d` (0: x, 1: y, 2: z)
struct Cell {
  int c[3];
  __host__ __device__ Cell shift(int d, int s) const {
    Cell r = *this;
    r.c[d] += s;
    return r;
  }
};

template <typename Real>
__host__ __device__ inline Real at(Array3DView<const Real> a, const Cell &p) {
  return a(p.c[0], p.c[1], p.c[2]);
}

/// @brief Inverse grid spacing of the cell `p` in direction `d`
template <typename Real>
__host__ __device__ inline Real inv_ds(const GridView<const Real> &grid, int d,
                                       const Cell &p) {
  return d == 0 ? grid.dxi[p.c[0]]
                : (d == 1 ? grid.dyi[p.c[1]] : grid.dzi[p.c[2]]);
}

/// @brief Resistive fluxes on the face between `pl` and `pl + e_d`.
/// @details G_md = eta (d_d B_m - d_m B_d) for m != d, and the energy flux
/// F = (1/4pi) sum_m B_m G_md. The normal derivative d_d B_m uses the two
/// cells adjacent to the face (compact), which avoids odd-even decoupling.
/// The tangential derivative d_m B_d is the average of the centered
/// differences in the two adjacent cells.
template <typename Real>
__host__ __device__ inline void
face_flux(const Array3DView<const Real> (&b)[3], Array3DView<const Real> eta,
          const GridView<const Real> &grid, const int (&st)[3], int d,
          const Cell &pl, Real (&gg)[3], Real &fe) {
  const Cell pr = pl.shift(d, st[d]);
  const Real eta_f = Real(0.5) * (at(eta, pl) + at(eta, pr));
  const Real idf = Real(0.5) * (inv_ds(grid, d, pl) + inv_ds(grid, d, pr));

  fe = Real(0);
  for (int m = 0; m < 3; ++m) {
    gg[m] = Real(0);
    if (m == d) {
      continue;
    }
    const Real normal = (at(b[m], pr) - at(b[m], pl)) * idf;
    const Real tangential =
        Real(0.25) *
        ((at(b[d], pl.shift(m, st[m])) - at(b[d], pl.shift(m, -st[m]))) *
             inv_ds(grid, m, pl) +
         (at(b[d], pr.shift(m, st[m])) - at(b[d], pr.shift(m, -st[m]))) *
             inv_ds(grid, m, pr));
    gg[m] = eta_f * (normal - tangential);
    fe += Real(0.5) * (at(b[m], pl) + at(b[m], pr)) * gg[m];
  }
  fe *= pii4<Real>;
}

}  // namespace impl_resistivity

/// @brief Add the resistive terms to the result of an update stage.
/// @details Evaluates -curl(eta curl B) and the corresponding energy flux
/// with `qq_argm`, and adds `dt` times them to `qq_rslt` (B and ei).
/// `eta` is the magnetic diffusivity at cell centers.
/// Ghost cells of `qq_argm` and `eta`, including edges and corners, must be
/// filled.
template <typename Real, typename Backend>
void add_resistive_terms(Backend btag, const Real dt,
                         const GridView<const Real> &grid,
                         Array3DView<const Real> eta,
                         const FieldsView<const Real> &qq_argm,
                         const FieldsView<Real> &qq_rslt) {
  using impl_resistivity::Cell;
  using impl_resistivity::face_flux;
  using impl_resistivity::inv_ds;

  Range3D range{{grid.i_margin, grid.i_total - grid.i_margin},
                {grid.j_margin, grid.j_total - grid.j_margin},
                {grid.k_margin, grid.k_total - grid.k_margin}};

  for_each(
      btag, range, MISO_LAMBDA(int i, int j, int k) {
        const Array3DView<const Real> b[3] = {qq_argm.bx, qq_argm.by, qq_argm.bz};
        const int st[3] = {grid.is, grid.js, grid.ks};
        const Cell p{{i, j, k}};

        Real db[3] = {Real(0), Real(0), Real(0)};
        Real de = Real(0);
        for (int d = 0; d < 3; ++d) {
          if (st[d] == 0) {
            continue;
          }
          Real gp[3], gm[3], fp, fm;
          face_flux(b, eta, grid, st, d, p, gp, fp);
          face_flux(b, eta, grid, st, d, p.shift(d, -st[d]), gm, fm);
          const Real dsi = inv_ds(grid, d, p);
          for (int m = 0; m < 3; ++m) {
            db[m] += (gp[m] - gm[m]) * dsi;
          }
          de += (fp - fm) * dsi;
        }

        const Real bx0 = qq_rslt.bx(i, j, k);
        const Real by0 = qq_rslt.by(i, j, k);
        const Real bz0 = qq_rslt.bz(i, j, k);
        const Real bx1 = bx0 + dt * db[0];
        const Real by1 = by0 + dt * db[1];
        const Real bz1 = bz0 + dt * db[2];
        qq_rslt.bx(i, j, k) = bx1;
        qq_rslt.by(i, j, k) = by1;
        qq_rslt.bz(i, j, k) = bz1;

        // Total energy gains dt*de; the change of the magnetic energy is
        // subtracted to obtain the internal energy (Joule heating).
        const Real dmag = pii8<Real> * ((bx1 * bx1 + by1 * by1 + bz1 * bz1) -
                                        (bx0 * bx0 + by0 * by0 + bz0 * bz0));
        qq_rslt.ei(i, j, k) += (dt * de - dmag) / qq_rslt.ro(i, j, k);
      });
}

/// @brief Explicit resistivity (magnetic diffusion) for MHD equations
template <typename Real, typename Backend> struct Resistivity {
  Grid<Real, Backend> &grid;
  /// @brief Magnetic diffusivity at cell centers (including ghost cells)
  Array3D<Real, Backend> eta;
  /// @brief Whether the resistive terms are applied
  bool enabled = false;
  /// @brief Upper limit of time step by the diffusion (global)
  Real dt_limit = Real(1.e10);
  /// @brief Safety factor of the diffusive time step.
  /// @details Stability of RK4 requires lambda*dt < 2.78 with
  /// lambda <= 4*eta*sum(1/ds^2).
  Real cfl_number;

  Resistivity(Config &config, Grid<Real, Backend> &grid)
      : grid(grid), eta(grid.i_total, grid.j_total, grid.k_total) {
    cfl_number = config["mhd"]["resistivity"]["cfl_number"].as<Real>();
  }

  /// @brief Set the magnetic diffusivity and enable the resistive terms.
  /// @param eta_h Diffusivity at all cells (including ghost cells) on host.
  void set_eta(const Array3D<Real, backend::Host> &eta_h) {
    eta.copy_from(eta_h);

    const auto eta_v = eta.const_view();
    const auto grid_v = grid.const_view();
    Range3D range{{grid.i_margin, grid.i_total - grid.i_margin},
                  {grid.j_margin, grid.j_total - grid.j_margin},
                  {grid.k_margin, grid.k_total - grid.k_margin}};
    // eta * sum(1/ds^2), maximum over the local domain
    const auto f = MISO_LAMBDA(int i, int j, int k) {
      const Real s = grid_v.dxi[i] * grid_v.dxi[i] +
                     grid_v.dyi[j] * grid_v.dyi[j] +
                     grid_v.dzi[k] * grid_v.dzi[k];
      return eta_v(i, j, k) * s;
    };
    const auto op = MISO_LAMBDA(Real a, Real b) { return util::max2(a, b); };
    const Real rate = reduce(Backend{}, range, Real(0), f, op);

    Real rate_g;
    MPI_Allreduce(&rate, &rate_g, 1, mpi::data_type<Real>(), MPI_MAX,
                  mpi::comm());
    if (rate_g < Real(0)) {
      throw std::runtime_error("Resistivity: eta must be non-negative.");
    }
    enabled = rate_g > Real(0);
    dt_limit = enabled ? cfl_number / rate_g : Real(1.e10);
  }

  /// @brief Add the resistive terms to `qq_rslt` (see add_resistive_terms)
  void apply(const Real dt, const Fields<Real, Backend> &qq_argm,
             Fields<Real, Backend> &qq_rslt) {
    if (!enabled) {
      return;
    }
    add_resistive_terms(Backend{}, dt, grid.const_view(), eta.const_view(),
                        qq_argm.const_view(), qq_rslt.view());
  }
};

}  // namespace mhd
}  // namespace miso
