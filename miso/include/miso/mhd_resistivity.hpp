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

/// @brief Resistive flux G_md = eta (d_d B_m - d_m B_d) on the face between
/// the cell (i, j, k) and (i + ni, j + nj, k + nk).
/// @details The face normal is the direction d given by the offsets
/// (ni, nj, nk) and the inverse spacing `dsi_n` of that axis. The tangential
/// direction m is given by (ti, tj, tk) and `dsi_t`, in the same way as the
/// artificial viscosity (e.g., (grid.is, 0, 0) and grid.dxi for x).
/// - d_d B_m (normal): compact difference of the two cells adjacent to the
///   face, which avoids odd-even decoupling.
/// - d_m B_d (tangential): average of the centered differences in the two
///   cells.
/// An inactive direction has zero offsets and zero inverse spacing, so that
/// its contribution vanishes.
template <typename Real>
__host__ __device__ inline Real resistive_flux(
    const Array3DView<const Real> &bm, const Array3DView<const Real> &bd,
    const Array3DView<const Real> &eta, const Real *dsi_n, int ni, int nj, int nk,
    const Real *dsi_t, int ti, int tj, int tk, int i, int j, int k) {
  // clang-format off
  const int ip = i + ni, jp = j + nj, kp = k + nk;
  const Real eta_f = Real(0.5) * (eta(i, j, k) + eta(ip, jp, kp));
  const Real dsi_f = Real(0.5) * (dsi_n[i * ni + j * nj + k * nk] +
                                  dsi_n[ip * ni + jp * nj + kp * nk]);
  const Real normal = (bm(ip, jp, kp) - bm(i, j, k)) * dsi_f;
  const Real tangential = Real(0.25) * (
      (bd(i  + ti, j  + tj, k  + tk) - bd(i  - ti, j  - tj, k  - tk)) * dsi_t[i  * ti + j  * tj + k  * tk]
    + (bd(ip + ti, jp + tj, kp + tk) - bd(ip - ti, jp - tj, kp - tk)) * dsi_t[ip * ti + jp * tj + kp * tk]);
  // clang-format on
  return eta_f * (normal - tangential);
}

/// @brief Explicit resistivity (magnetic diffusion) as a source term.
/// @details Adds -curl(eta curl B) to the induction equation and the
/// corresponding energy flux div((1/4pi) eta (curl B) x B) to the energy
/// equation (the Joule heating goes to the internal energy). In index
/// notation, dB_i/dt = d_j G_ij, dE/dt = d_j ((1/4pi) B_i G_ij),
/// G_ij = eta (d_j B_i - d_i B_j), discretized in the flux form with second
/// order accuracy (see resistive_flux).
///
/// Not included by <miso/mhd.hpp>; include <miso/mhd_resistivity.hpp>
/// explicitly.
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

  // clang-format off
  /// @brief x induction: d_y G_xy + d_z G_xz
  __host__ __device__ Real bx(const FieldsView<const Real> &qq, int i, int j,
                              int k) const {
    const int is = grid.is, js = grid.js, ks = grid.ks;
    return
      + (resistive_flux(qq.bx, qq.by, eta, grid.dyi, 0, js, 0, grid.dxi, is, 0, 0, i, j     , k)
       - resistive_flux(qq.bx, qq.by, eta, grid.dyi, 0, js, 0, grid.dxi, is, 0, 0, i, j - js, k)) * grid.dyi[j]
      + (resistive_flux(qq.bx, qq.bz, eta, grid.dzi, 0, 0, ks, grid.dxi, is, 0, 0, i, j, k     )
       - resistive_flux(qq.bx, qq.bz, eta, grid.dzi, 0, 0, ks, grid.dxi, is, 0, 0, i, j, k - ks)) * grid.dzi[k];
  }

  /// @brief y induction: d_x G_yx + d_z G_yz
  __host__ __device__ Real by(const FieldsView<const Real> &qq, int i, int j,
                              int k) const {
    const int is = grid.is, js = grid.js, ks = grid.ks;
    return
      + (resistive_flux(qq.by, qq.bx, eta, grid.dxi, is, 0, 0, grid.dyi, 0, js, 0, i     , j, k)
       - resistive_flux(qq.by, qq.bx, eta, grid.dxi, is, 0, 0, grid.dyi, 0, js, 0, i - is, j, k)) * grid.dxi[i]
      + (resistive_flux(qq.by, qq.bz, eta, grid.dzi, 0, 0, ks, grid.dyi, 0, js, 0, i, j, k     )
       - resistive_flux(qq.by, qq.bz, eta, grid.dzi, 0, 0, ks, grid.dyi, 0, js, 0, i, j, k - ks)) * grid.dzi[k];
  }

  /// @brief z induction: d_x G_zx + d_y G_zy
  __host__ __device__ Real bz(const FieldsView<const Real> &qq, int i, int j,
                              int k) const {
    const int is = grid.is, js = grid.js, ks = grid.ks;
    return
      + (resistive_flux(qq.bz, qq.bx, eta, grid.dxi, is, 0, 0, grid.dzi, 0, 0, ks, i     , j, k)
       - resistive_flux(qq.bz, qq.bx, eta, grid.dxi, is, 0, 0, grid.dzi, 0, 0, ks, i - is, j, k)) * grid.dxi[i]
      + (resistive_flux(qq.bz, qq.by, eta, grid.dyi, 0, js, 0, grid.dzi, 0, 0, ks, i, j     , k)
       - resistive_flux(qq.bz, qq.by, eta, grid.dyi, 0, js, 0, grid.dzi, 0, 0, ks, i, j - js, k)) * grid.dyi[j];
  }

  /// @brief Energy: divergence of the energy flux (1/4pi) B_m G_md
  __host__ __device__ Real ei(const FieldsView<const Real> &qq, int i, int j,
                              int k) const {
    const int is = grid.is, js = grid.js, ks = grid.ks;
    return pii4<Real> * (
      + (energy_flux_x(qq, i, j, k) - energy_flux_x(qq, i - is, j, k)) * grid.dxi[i]
      + (energy_flux_y(qq, i, j, k) - energy_flux_y(qq, i, j - js, k)) * grid.dyi[j]
      + (energy_flux_z(qq, i, j, k) - energy_flux_z(qq, i, j, k - ks)) * grid.dzi[k]);
  }
  // clang-format on

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
  // clang-format off
  /// @brief sum_m B_m G_mx on the x face between (i, j, k) and (i + is, j, k)
  __host__ __device__ Real energy_flux_x(const FieldsView<const Real> &qq,
                                         int i, int j, int k) const {
    const int is = grid.is, js = grid.js, ks = grid.ks;
    return
      + Real(0.5) * (qq.by(i, j, k) + qq.by(i + is, j, k))
        * resistive_flux(qq.by, qq.bx, eta, grid.dxi, is, 0, 0, grid.dyi, 0, js, 0, i, j, k)
      + Real(0.5) * (qq.bz(i, j, k) + qq.bz(i + is, j, k))
        * resistive_flux(qq.bz, qq.bx, eta, grid.dxi, is, 0, 0, grid.dzi, 0, 0, ks, i, j, k);
  }

  /// @brief sum_m B_m G_my on the y face between (i, j, k) and (i, j + js, k)
  __host__ __device__ Real energy_flux_y(const FieldsView<const Real> &qq,
                                         int i, int j, int k) const {
    const int is = grid.is, js = grid.js, ks = grid.ks;
    return
      + Real(0.5) * (qq.bx(i, j, k) + qq.bx(i, j + js, k))
        * resistive_flux(qq.bx, qq.by, eta, grid.dyi, 0, js, 0, grid.dxi, is, 0, 0, i, j, k)
      + Real(0.5) * (qq.bz(i, j, k) + qq.bz(i, j + js, k))
        * resistive_flux(qq.bz, qq.by, eta, grid.dyi, 0, js, 0, grid.dzi, 0, 0, ks, i, j, k);
  }

  /// @brief sum_m B_m G_mz on the z face between (i, j, k) and (i, j, k + ks)
  __host__ __device__ Real energy_flux_z(const FieldsView<const Real> &qq,
                                         int i, int j, int k) const {
    const int is = grid.is, js = grid.js, ks = grid.ks;
    return
      + Real(0.5) * (qq.bx(i, j, k) + qq.bx(i, j, k + ks))
        * resistive_flux(qq.bx, qq.bz, eta, grid.dzi, 0, 0, ks, grid.dxi, is, 0, 0, i, j, k)
      + Real(0.5) * (qq.by(i, j, k) + qq.by(i, j, k + ks))
        * resistive_flux(qq.by, qq.bz, eta, grid.dzi, 0, 0, ks, grid.dyi, 0, js, 0, i, j, k);
  }
  // clang-format on
};

}  // namespace mhd
}  // namespace miso
