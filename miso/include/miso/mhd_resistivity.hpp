#pragma once

#include "array3d.hpp"
#include "config.hpp"
#include "constants.hpp"
#include "execution.hpp"
#include "grid.hpp"
#include "mhd_fields.hpp"
#include "mpi_util.hpp"
#include "utility.hpp"

namespace miso {
namespace mhd {

namespace impl_resistivity {

/// @brief Cell index (i, j, k) that can be shifted in direction d
struct Cell {
  int c[3];
  __host__ __device__ Cell shift(int d, int s) const {
    Cell r = *this;
    r.c[d] += s;
    return r;
  }
};

template <typename Real>
__host__ __device__ inline Real at(const Array3DView<const Real> &a,
                                   const Cell &p) {
  return a(p.c[0], p.c[1], p.c[2]);
}

/// @brief Inverse grid spacing of the cell `p` in direction `d`
template <typename Real>
__host__ __device__ inline Real inv_ds(const GridView<const Real> &grid, int d,
                                       const Cell &p) {
  return d == 0 ? grid.dxi[p.c[0]]
                : (d == 1 ? grid.dyi[p.c[1]] : grid.dzi[p.c[2]]);
}

/// @brief G_md = eta (d_d B_m - d_m B_d) on the face between `pl` and
/// `pl + e_d` (m != d).
/// @details The normal derivative d_d B_m uses the two cells adjacent to the
/// face (compact), which avoids odd-even decoupling. The tangential
/// derivative d_m B_d is the average of the centered differences in the two
/// adjacent cells.
template <typename Real>
__host__ __device__ inline Real
face_flux(const Array3DView<const Real> (&b)[3],
          const Array3DView<const Real> &eta, const GridView<const Real> &grid,
          const int (&st)[3], int d, int m, const Cell &pl) {
  const Cell pr = pl.shift(d, st[d]);
  const Real eta_f = Real(0.5) * (at(eta, pl) + at(eta, pr));
  const Real idf = Real(0.5) * (inv_ds(grid, d, pl) + inv_ds(grid, d, pr));
  const Real normal = (at(b[m], pr) - at(b[m], pl)) * idf;
  const Real tangential =
      Real(0.25) *
      ((at(b[d], pl.shift(m, st[m])) - at(b[d], pl.shift(m, -st[m]))) *
           inv_ds(grid, m, pl) +
       (at(b[d], pr.shift(m, st[m])) - at(b[d], pr.shift(m, -st[m]))) *
           inv_ds(grid, m, pr));
  return eta_f * (normal - tangential);
}

/// @brief Energy flux (1/4pi) sum_m B_m G_md on the face between `pl` and
/// `pl + e_d` (B is averaged to the face).
template <typename Real>
__host__ __device__ inline Real face_energy_flux(
    const Array3DView<const Real> (&b)[3], const Array3DView<const Real> &eta,
    const GridView<const Real> &grid, const int (&st)[3], int d, const Cell &pl) {
  const Cell pr = pl.shift(d, st[d]);
  Real fe = Real(0);
  for (int m = 0; m < 3; ++m) {
    if (m == d) {
      continue;
    }
    fe += Real(0.5) * (at(b[m], pl) + at(b[m], pr)) *
          face_flux(b, eta, grid, st, d, m, pl);
  }
  return pii4<Real> * fe;
}

}  // namespace impl_resistivity

/// @brief Explicit resistivity (magnetic diffusion) as a source term.
/// @details Adds -curl(eta curl B) to the induction equation and the
/// corresponding energy flux div((1/4pi) eta (curl B) x B) to the energy
/// equation (the Joule heating goes to the internal energy). In index
/// notation, dB_i/dt = d_j G_ij, dE/dt = d_j ((1/4pi) B_i G_ij),
/// G_ij = eta (d_j B_i - d_i B_j), discretized in the flux form with second
/// order accuracy.
///
/// This struct only holds views: the diffusivity array `eta` (at cell
/// centers, including ghost cells) is owned by the caller and must outlive
/// this object. Use it as `src` of a model, or call its member functions from
/// a user-defined source term to combine it with other sources.
/// @code
/// struct Model : public mhd::ModelBase<Model, Real, Backend> {
///   ...
///   Array3D<Real, Backend> eta;
///   mhd::ResistiveSource<Real, Backend> src;
///   Model(Config &config)
///       : ModelBase(config), ..., eta(make_eta(...)),
///         src(config, mhd.grid, eta) {}
/// };
/// @endcode
/// @note Ghost cells of the MHD fields including edges and corners are used.
template <typename Real, typename Backend> struct ResistiveSource {
  /// @brief Grid (of the backend)
  GridView<const Real> grid;
  /// @brief Magnetic diffusivity at cell centers (including ghost cells)
  Array3DView<const Real> eta;
  /// @brief Safety factor of the diffusive time step.
  /// @details Stability of RK4 requires lambda*dt < 2.78 with
  /// lambda <= 4*eta*sum(1/ds^2), i.e., cfl_number < 0.69.
  Real cfl_number;

  ResistiveSource(const Grid<Real, Backend> &grid_,
                  const Array3D<Real, Backend> &eta_, Real cfl_number_)
      : grid(grid_.const_view()), eta(eta_.const_view()),
        cfl_number(cfl_number_) {}

  /// @brief Construct with `mhd.resistivity.cfl_number` in the config
  ResistiveSource(const Config &config, const Grid<Real, Backend> &grid_,
                  const Array3D<Real, Backend> &eta_)
      : ResistiveSource(
            grid_, eta_,
            config["mhd"]["resistivity"]["cfl_number"].template as<Real>()) {}

  __host__ __device__ Real vx(const FieldsView<const Real> &, int, int,
                              int) const {
    return Real(0);
  }
  __host__ __device__ Real vy(const FieldsView<const Real> &, int, int,
                              int) const {
    return Real(0);
  }
  __host__ __device__ Real vz(const FieldsView<const Real> &, int, int,
                              int) const {
    return Real(0);
  }
  __host__ __device__ Real bx(const FieldsView<const Real> &qq, int i, int j,
                              int k) const {
    return induction(qq, 0, i, j, k);
  }
  __host__ __device__ Real by(const FieldsView<const Real> &qq, int i, int j,
                              int k) const {
    return induction(qq, 1, i, j, k);
  }
  __host__ __device__ Real bz(const FieldsView<const Real> &qq, int i, int j,
                              int k) const {
    return induction(qq, 2, i, j, k);
  }

  /// @brief Divergence of the resistive energy flux
  __host__ __device__ Real ei(const FieldsView<const Real> &qq, int i, int j,
                              int k) const {
    using namespace impl_resistivity;
    const Array3DView<const Real> b[3] = {qq.bx, qq.by, qq.bz};
    const int st[3] = {grid.is, grid.js, grid.ks};
    const Cell p{{i, j, k}};
    Real de = Real(0);
    for (int d = 0; d < 3; ++d) {
      if (st[d] == 0) {
        continue;
      }
      de += (face_energy_flux(b, eta, grid, st, d, p) -
             face_energy_flux(b, eta, grid, st, d, p.shift(d, -st[d]))) *
            inv_ds(grid, d, p);
    }
    return de;
  }

  /// @brief Upper limit of the time step: cfl_number / max(eta*sum(1/ds^2))
  Real dt_limit() const {
    const auto eta_v = eta;  // device lambdas must not capture `this`
    const auto grid_v = grid;
    Range3D range{{grid.i_margin, grid.i_total - grid.i_margin},
                  {grid.j_margin, grid.j_total - grid.j_margin},
                  {grid.k_margin, grid.k_total - grid.k_margin}};
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
    return rate_g > Real(0) ? cfl_number / rate_g : Real(1.e10);
  }

private:
  /// @brief (d/dt B_m) = sum_{d != m} d_d G_md
  __host__ __device__ Real induction(const FieldsView<const Real> &qq, int m,
                                     int i, int j, int k) const {
    using namespace impl_resistivity;
    const Array3DView<const Real> b[3] = {qq.bx, qq.by, qq.bz};
    const int st[3] = {grid.is, grid.js, grid.ks};
    const Cell p{{i, j, k}};
    Real db = Real(0);
    for (int d = 0; d < 3; ++d) {
      if (d == m || st[d] == 0) {
        continue;
      }
      db += (face_flux(b, eta, grid, st, d, m, p) -
             face_flux(b, eta, grid, st, d, m, p.shift(d, -st[d]))) *
            inv_ds(grid, d, p);
    }
    return db;
  }
};

}  // namespace mhd
}  // namespace miso
