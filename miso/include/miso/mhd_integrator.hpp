#pragma once

#include <type_traits>
#include <utility>

#include "array3d.hpp"
#include "constants.hpp"
#include "cuda_compat.hpp"
#include "env.hpp"
#include "execution.hpp"
#include "mhd_artificial_viscosity.hpp"
#include "mhd_fields.hpp"
#include "mhd_halo_exchange.hpp"

namespace miso {
namespace mhd {

// Source terms are optional. A Source type may define any of the following
// member functions, each returning the source term of the equation for the
// named variable at cell (i, j, k):
//   Real ro(FieldsView<const Real> qq, int i, int j, int k) const;  // mass
//   Real vx(...), vy(...), vz(...)  // momentum (force per unit volume)
//   Real bx(...), by(...), bz(...)  // induction equation
//   Real ei(...)                    // total energy (per unit volume and time)
// A term that is not defined is not computed (treated as 0). A member with
// the same name that cannot be called in this way (e.g., not const, or
// taking FieldsView<Real>) is a compile error, not silently ignored.
// With the CUDA backend, the member functions must be __host__ __device__.
// clang-format off
#define MISO_MHD_DEFINE_SOURCE_TERM(name)                                      \
  namespace impl_source {                                                      \
  template <class Source, class = void>                                        \
  struct has_member_##name : std::false_type {};                               \
  template <class Source>                                                      \
  struct has_member_##name<Source, std::void_t<decltype(&Source::name)>>       \
      : std::true_type {};                                                     \
  template <class Source, class Real, class = void>                            \
  struct has_##name : std::false_type {};                                      \
  template <class Source, class Real>                                          \
  struct has_##name<Source, Real,                                              \
                    std::void_t<decltype(std::declval<const Source &>().name(  \
                        std::declval<FieldsView<const Real>>(), 0, 0, 0))>>    \
      : std::true_type {};                                                     \
  }                                                                            \
  /** @brief Source term of `name` (0 if the Source does not define it) */     \
  template <typename Real, typename Source>                                    \
  __host__ __device__ inline Real source_##name(                               \
      const Source &src, const FieldsView<const Real> &qq, int i, int j,       \
      int k) {                                                                 \
    static_assert(!impl_source::has_member_##name<Source>::value ||            \
                      impl_source::has_##name<Source, Real>::value,            \
                  "Source::" #name " must be callable as " #name               \
                  "(FieldsView<const Real>, int, int, int) const");            \
    if constexpr (impl_source::has_##name<Source, Real>::value) {              \
      return src.name(qq, i, j, k);                                            \
    } else {                                                                   \
      return Real(0);                                                          \
    }                                                                          \
  }
MISO_MHD_DEFINE_SOURCE_TERM(ro)
MISO_MHD_DEFINE_SOURCE_TERM(vx)
MISO_MHD_DEFINE_SOURCE_TERM(vy)
MISO_MHD_DEFINE_SOURCE_TERM(vz)
MISO_MHD_DEFINE_SOURCE_TERM(bx)
MISO_MHD_DEFINE_SOURCE_TERM(by)
MISO_MHD_DEFINE_SOURCE_TERM(bz)
MISO_MHD_DEFINE_SOURCE_TERM(ei)
#undef MISO_MHD_DEFINE_SOURCE_TERM
// clang-format on

/// @brief Calculate 4th order space-centered derivative for qq
template <typename Real>
__host__ __device__ inline Real
space_centered_4th(Array3DView<const Real> qq, Real dxyzi, int i, int j, int k,
                   int is, int js, int ks) {
  // clang-format off
  return (
    -     qq(i + 2*is, j + 2*js, k + 2*ks)
    + 8.0*qq(i +   is, j +   js, k +   ks)
    - 8.0*qq(i -   is, j -   js, k -   ks)
    +     qq(i - 2*is, j - 2*js, k - 2*ks)
  )*inv12<Real>*dxyzi;
  // clang-format on
};

/// @brief Calculate 4th order space-centered derivative for qq1*qq2
template <typename Real>
__host__ __device__ inline Real
space_centered_4th(Array3DView<const Real> qq1, Array3DView<const Real> qq2,
                   Real dxyzi, int i, int j, int k, int is, int js, int ks) {
  // clang-format off
  return (
    -     qq1(i + 2*is, j + 2*js, k + 2*ks)*qq2(i + 2*is, j + 2*js, k + 2*ks)
    + 8.0*qq1(i +   is, j +   js, k +   ks)*qq2(i +   is, j +   js, k +   ks)
    - 8.0*qq1(i -   is, j -   js, k -   ks)*qq2(i -   is, j -   js, k -   ks)
    +     qq1(i - 2*is, j - 2*js, k - 2*ks)*qq2(i - 2*is, j - 2*js, k - 2*ks)
  )*inv12<Real>*dxyzi;
  // clang-format on
};

/// @brief Calculate 4th order space-centered derivative for qq1*qq2*qq3
template <typename Real>
__host__ __device__ inline Real
space_centered_4th(Array3DView<const Real> qq1, Array3DView<const Real> qq2,
                   Array3DView<const Real> qq3, Real dxyzi, int i, int j, int k,
                   int is, int js, int ks) {
  // clang-format off
  return (
    -     qq1(i + 2*is, j + 2*js, k + 2*ks)*qq2(i + 2*is, j + 2*js, k + 2*ks)*qq3(i + 2*is, j + 2*js, k + 2*ks)
    + 8.0*qq1(i +   is, j +   js, k +   ks)*qq2(i +   is, j +   js, k +   ks)*qq3(i +   is, j +   js, k +   ks)
    - 8.0*qq1(i -   is, j -   js, k -   ks)*qq2(i -   is, j -   js, k -   ks)*qq3(i -   is, j -   js, k -   ks)
    +     qq1(i - 2*is, j - 2*js, k - 2*ks)*qq2(i - 2*is, j - 2*js, k - 2*ks)*qq3(i - 2*is, j - 2*js, k - 2*ks)
  )*inv12<Real>*dxyzi;
  // clang-format on
};

/// @brief Right-hand sides of the 4th order space-centered scheme at a cell.
/// @details Written once for all backends: every member function is
/// __host__ __device__ and called from for_each. The struct holds views only,
/// so it is cheap to copy into a CUDA kernel. Each function writes one
/// variable of qq_rslt at (i, j, k); vx, vy, vz and ei read qq_rslt at the
/// same cell, so call ro before them, and ro, vx, ..., bz before ei.
template <typename Real, typename Source> struct Sc4Kernel {
  FieldsView<const Real> qq_orgn, qq_argm;
  FieldsView<Real> qq_rslt;
  Array3DView<const Real> pr, bb, ht, vb;
  Source src;
  GridView<const Real> grid;
  Real dt, ch_divb_square, tau_divb;

  // clang-format off
  /// @brief Equation of continuity
  __host__ __device__ void ro(int i, int j, int k) const {
    qq_rslt.ro(i, j, k) = qq_orgn.ro(i, j, k) + dt * (
        - space_centered_4th(qq_argm.ro, qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0)
        - space_centered_4th(qq_argm.ro, qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0)
        - space_centered_4th(qq_argm.ro, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks)
        + source_ro(src, qq_argm, i, j, k));
  }

  /// @brief x equation of motion
  __host__ __device__ void vx(int i, int j, int k) const {
    qq_rslt.vx(i, j, k) =
        (qq_orgn.ro(i, j, k) * qq_orgn.vx(i, j, k) +
         dt *
             (- space_centered_4th(qq_argm.ro, qq_argm.vx, qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0)
              - space_centered_4th(qq_argm.ro, qq_argm.vx, qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0)
              - space_centered_4th(qq_argm.ro, qq_argm.vx, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks)
              - space_centered_4th(pr, grid.dxi[i], i, j, k, grid.is, 0, 0)
              - space_centered_4th(bb, grid.dxi[i], i, j, k, grid.is, 0, 0) * pii8<Real>
              + pii4<Real> * (+ space_centered_4th(qq_argm.bx, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0)
                              + space_centered_4th(qq_argm.bx, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0)
                              + space_centered_4th(qq_argm.bx, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks))
              + source_vx(src, qq_argm, i, j, k)
              )) / qq_rslt.ro(i, j, k);
  }

  /// @brief y equation of motion
  __host__ __device__ void vy(int i, int j, int k) const {
    qq_rslt.vy(i, j, k) =
        (qq_orgn.ro(i, j, k) * qq_orgn.vy(i, j, k) +
         dt *
             (- space_centered_4th(qq_argm.ro, qq_argm.vy, qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0)
              - space_centered_4th(qq_argm.ro, qq_argm.vy, qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0)
              - space_centered_4th(qq_argm.ro, qq_argm.vy, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks)
              - space_centered_4th(pr, grid.dyi[j], i, j, k, 0, grid.js, 0)
              - space_centered_4th(bb, grid.dyi[j], i, j, k, 0, grid.js, 0) * pii8<Real>
              + pii4<Real> * (+ space_centered_4th(qq_argm.by, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0)
                              + space_centered_4th(qq_argm.by, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0)
                              + space_centered_4th(qq_argm.by, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks))
              + source_vy(src, qq_argm, i, j, k)
              )) / qq_rslt.ro(i, j, k);
  }

  /// @brief z equation of motion
  __host__ __device__ void vz(int i, int j, int k) const {
    qq_rslt.vz(i, j, k) =
        (qq_orgn.ro(i, j, k) * qq_orgn.vz(i, j, k) +
         dt *
             ( - space_centered_4th(qq_argm.ro, qq_argm.vz, qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0)
               - space_centered_4th(qq_argm.ro, qq_argm.vz, qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0)
               - space_centered_4th(qq_argm.ro, qq_argm.vz, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks)
               - space_centered_4th(pr, grid.dzi[k], i, j, k, 0, 0, grid.ks)
               - space_centered_4th(bb, grid.dzi[k], i, j, k, 0, 0, grid.ks) * pii8<Real>
               + pii4<Real> * (+ space_centered_4th(qq_argm.bz, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0)
                               + space_centered_4th(qq_argm.bz, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0)
                               + space_centered_4th(qq_argm.bz, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks))
               + source_vz(src, qq_argm, i, j, k)
            )) / qq_rslt.ro(i, j, k);
  }

  /// @brief x magnetic induction
  __host__ __device__ void bx(int i, int j, int k) const {
    qq_rslt.bx(i, j, k) =
        qq_orgn.bx(i, j, k) +
        dt * (- space_centered_4th(qq_argm.vy, qq_argm.bx, grid.dyi[j], i, j, k, 0, grid.js, 0)
              - space_centered_4th(qq_argm.vz, qq_argm.bx, grid.dzi[k], i, j, k, 0, 0, grid.ks)
              + space_centered_4th(qq_argm.vx, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0)
              + space_centered_4th(qq_argm.vx, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks)
              - space_centered_4th(qq_argm.ph, grid.dxi[i], i, j, k, grid.is, 0, 0)
              + source_bx(src, qq_argm, i, j, k));
  }

  /// @brief y magnetic induction
  __host__ __device__ void by(int i, int j, int k) const {
    qq_rslt.by(i, j, k) =
        qq_orgn.by(i, j, k) +
        dt * (- space_centered_4th(qq_argm.vx, qq_argm.by, grid.dxi[i], i, j, k, grid.is, 0, 0)
              - space_centered_4th(qq_argm.vz, qq_argm.by, grid.dzi[k], i, j, k, 0, 0, grid.ks)
              + space_centered_4th(qq_argm.vy, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0)
              + space_centered_4th(qq_argm.vy, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks)
              - space_centered_4th(qq_argm.ph, grid.dyi[j], i, j, k, 0, grid.js, 0)
              + source_by(src, qq_argm, i, j, k));
  }

  /// @brief z magnetic induction
  __host__ __device__ void bz(int i, int j, int k) const {
    qq_rslt.bz(i, j, k) =
        qq_orgn.bz(i, j, k) +
        dt * (- space_centered_4th(qq_argm.vx, qq_argm.bz, grid.dxi[i], i, j, k, grid.is, 0, 0)
              - space_centered_4th(qq_argm.vy, qq_argm.bz, grid.dyi[j], i, j, k, 0, grid.js, 0)
              + space_centered_4th(qq_argm.vz, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0)
              + space_centered_4th(qq_argm.vz, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0)
              - space_centered_4th(qq_argm.ph, grid.dzi[k], i, j, k, 0, 0, grid.ks)
              + source_bz(src, qq_argm, i, j, k));
  }

  /// @brief div B cleaning
  __host__ __device__ void ph(int i, int j, int k) const {
    qq_rslt.ph(i, j, k) =
        (qq_orgn.ph(i, j, k) +
         dt * ch_divb_square *
             (- space_centered_4th(qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0)
              - space_centered_4th(qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0)
              - space_centered_4th(qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks))) *
        util::exp(-dt / tau_divb);
  }

  /// @brief Energy equation
  /// @details Et: total energy per unit volume, ei: internal energy per unit
  /// mass
  __host__ __device__ void ei(int i, int j, int k) const {
    const Real Et =
        +qq_orgn.ro(i, j, k) * qq_orgn.ei(i, j, k) +
        0.5 * qq_orgn.ro(i, j, k) *
            (+ qq_orgn.vx(i, j, k) * qq_orgn.vx(i, j, k)
             + qq_orgn.vy(i, j, k) * qq_orgn.vy(i, j, k)
             + qq_orgn.vz(i, j, k) * qq_orgn.vz(i, j, k)) +
        pii8<Real> *
            (+ qq_orgn.bx(i, j, k) * qq_orgn.bx(i, j, k)
             + qq_orgn.by(i, j, k) * qq_orgn.by(i, j, k)
             + qq_orgn.bz(i, j, k) * qq_orgn.bz(i, j, k));

    qq_rslt.ei(i, j, k) =
        (Et +
         dt * (- space_centered_4th(ht, qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0)
               - space_centered_4th(ht, qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0)
               - space_centered_4th(ht, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks)
               + pii4<Real> * (+ space_centered_4th(vb, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0)
                               + space_centered_4th(vb, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0)
                               + space_centered_4th(vb, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks)))
         - 0.5 * qq_rslt.ro(i, j, k) *
             (+ qq_rslt.vx(i, j, k) * qq_rslt.vx(i, j, k)
              + qq_rslt.vy(i, j, k) * qq_rslt.vy(i, j, k)
              + qq_rslt.vz(i, j, k) * qq_rslt.vz(i, j, k)) -
         pii8<Real> *
             (+ qq_rslt.bx(i, j, k) * qq_rslt.bx(i, j, k)
              + qq_rslt.by(i, j, k) * qq_rslt.by(i, j, k)
              + qq_rslt.bz(i, j, k) * qq_rslt.bz(i, j, k))
         + dt * source_ei(src, qq_argm, i, j, k)
        ) / qq_rslt.ro(i, j, k);
  }
  // clang-format on
};

/// @brief Class for time integration of MHD equations
/// @details One implementation for all backends. Loops are written with
/// for_each / reduce of the backend, so that the same code runs on the host
/// and on CUDA devices.
template <typename Real, typename Backend> struct Integrator {
  /// @brief Spatial grid
  Grid<Real, Backend> &grid;
  /// @brief Workspace
  Fields<Real, Backend> qq_argm, qq_rslt;

  /// @brief Halo exchanger
  HaloExchanger<Real, Backend> halo_exchanger;
  /// @brief Artificial viscosity for MHD equations
  ArtificialViscosity<Real, Backend> artdiff;

  /// @brief Workspace for timestep calculation
  ReduceWorkspace<Real, Backend> cfl_workspace;

  /// @brief gas pressure
  Array3D<Real, Backend> pr;
  /// @brief magnetic field strength bx*bx + by*by + bz*bz
  Array3D<Real, Backend> bb;
  /// @brief enthalpy + 2*magnetic energy + kinetic energy
  Array3D<Real, Backend> ht;
  /// @brief inner product of velocity and magnetic field vx*bx + vy*by + vz*bz
  Array3D<Real, Backend> vb;
  /// @brief speed of sound
  Array3D<Real, Backend> cs;

  /// @brief CFL number
  Real cfl_number;
  /// @brief propagation speed fo divergence B
  Real ch_divb;
  /// @brief square of ch_divb;
  Real ch_divb_square;
  /// @brief damping time scape for divergence B
  Real tau_divb;

  /// @brief Constructor
  Integrator(Config &config, Grid<Real, Backend> &grid,
             ExecContext<Real, Backend> &exec_ctx)
      : grid(grid), qq_argm(grid), qq_rslt(grid), halo_exchanger(grid, exec_ctx),
        artdiff(config, grid, exec_ctx),
        pr(grid.i_total, grid.j_total, grid.k_total),
        bb(grid.i_total, grid.j_total, grid.k_total),
        ht(grid.i_total, grid.j_total, grid.k_total),
        vb(grid.i_total, grid.j_total, grid.k_total),
        cs(grid.i_total, grid.j_total, grid.k_total) {
    cfl_number = config["mhd"]["cfl_number"].as<Real>();
  }

  /// @brief Range of all cells including ghost cells
  Range3D whole_range() const {
    return {{0, grid.i_total}, {0, grid.j_total}, {0, grid.k_total}};
  }

  /// @brief Range of cells excluding ghost cells
  Range3D inner_range() const {
    return {{grid.i_margin, grid.i_total - grid.i_margin},
            {grid.j_margin, grid.j_total - grid.j_margin},
            {grid.k_margin, grid.k_total - grid.k_margin}};
  }

  /// @brief Update MHD equations using 4th order space-centered scheme
  template <typename EOS, typename Source>
  void update_sc4(const Real dt, const EOS &eos, const Source &src,
                  const Fields<Real, Backend> &qq_orgn,
                  const Fields<Real, Backend> &qq_argm,
                  Fields<Real, Backend> &qq_rslt) {
    eos.gas_pressure(Backend{}, qq_argm.const_view(), pr.view());

    const auto c_qq = qq_argm.const_view();
    const auto c_pr = pr.const_view();
    const auto bb_v = bb.view();
    const auto ht_v = ht.view();
    const auto vb_v = vb.view();
    for_each(
        Backend{}, whole_range(), MISO_LAMBDA(int i, int j, int k) {
          // clang-format off
      // squared magnetic strength
      bb_v(i, j, k) = c_qq.bx(i, j, k) * c_qq.bx(i, j, k) +
                      c_qq.by(i, j, k) * c_qq.by(i, j, k) +
                      c_qq.bz(i, j, k) * c_qq.bz(i, j, k);
      // enthalpy + 2*magnetic energy + kinetic energy
      ht_v(i, j, k) =
          +c_qq.ro(i, j, k) * c_qq.ei(i, j, k) +
          c_pr(i, j, k) + bb_v(i, j, k) * pii4<Real> +
          0.5 * c_qq.ro(i, j, k) *
              (+ c_qq.vx(i, j, k) * c_qq.vx(i, j, k)
               + c_qq.vy(i, j, k) * c_qq.vy(i, j, k)
               + c_qq.vz(i, j, k) * c_qq.vz(i, j, k));
      // v dot b
      vb_v(i, j, k) =
          + c_qq.vx(i, j, k) * c_qq.bx(i, j, k)
          + c_qq.vy(i, j, k) * c_qq.by(i, j, k)
          + c_qq.vz(i, j, k) * c_qq.bz(i, j, k);
          // clang-format on
        });

    // Members are copied into the kernel (device lambdas must not capture
    // `this`).
    const Sc4Kernel<Real, Source> kern{qq_orgn.const_view(), c_qq,
                                       qq_rslt.view(),       c_pr,
                                       bb.const_view(),      ht.const_view(),
                                       vb.const_view(),      src,
                                       grid.const_view(),    dt,
                                       ch_divb_square,       tau_divb};

    // One launch per equation, as the former CUDA implementation. Fusing them
    // into one for_each (as the former host implementation) is a one-line
    // change; which is faster depends on the backend (see the PR).
    const auto range = inner_range();
    for_each(
        Backend{}, range, MISO_LAMBDA(int i, int j, int k) { kern.ro(i, j, k); });
    for_each(
        Backend{}, range, MISO_LAMBDA(int i, int j, int k) { kern.vx(i, j, k); });
    for_each(
        Backend{}, range, MISO_LAMBDA(int i, int j, int k) { kern.vy(i, j, k); });
    for_each(
        Backend{}, range, MISO_LAMBDA(int i, int j, int k) { kern.vz(i, j, k); });
    for_each(
        Backend{}, range, MISO_LAMBDA(int i, int j, int k) { kern.bx(i, j, k); });
    for_each(
        Backend{}, range, MISO_LAMBDA(int i, int j, int k) { kern.by(i, j, k); });
    for_each(
        Backend{}, range, MISO_LAMBDA(int i, int j, int k) { kern.bz(i, j, k); });
    for_each(
        Backend{}, range, MISO_LAMBDA(int i, int j, int k) { kern.ph(i, j, k); });
    for_each(
        Backend{}, range, MISO_LAMBDA(int i, int j, int k) { kern.ei(i, j, k); });
  }

  /// @brief Apply boundary condition and halo exchange
  template <typename BoundaryCondition>
  void apply_boundary_condition(const BoundaryCondition &bc,
                                Fields<Real, Backend> &qq) {
    bc.apply(qq.view(), grid.const_view());
    fence(Backend{});  // May not be necessary
    halo_exchanger.apply(qq);
  }

  /// @brief Runge-Kutta 4th order time integration step
  template <typename EOS, typename BoundaryCondition, typename Source>
  void runge_kutta_4step(const Real dt, const EOS &eos,
                         const BoundaryCondition &bc, const Source &src,
                         Fields<Real, Backend> &qq) {
    update_sc4(dt / 4.0, eos, src, qq, qq, qq_rslt);
    qq_argm.copy_from(qq_rslt);
    apply_boundary_condition(bc, qq_argm);

    update_sc4(dt / 3.0, eos, src, qq, qq_argm, qq_rslt);
    qq_argm.copy_from(qq_rslt);
    apply_boundary_condition(bc, qq_argm);

    update_sc4(dt / 2.0, eos, src, qq, qq_argm, qq_rslt);
    qq_argm.copy_from(qq_rslt);
    apply_boundary_condition(bc, qq_argm);

    update_sc4(dt, eos, src, qq, qq_argm, qq_rslt);
    qq.copy_from(qq_rslt);
    apply_boundary_condition(bc, qq);
  }

  /// @brief Apply artificial viscosity
  template <typename EOS, typename BoundaryCondition>
  void apply_artificial_viscosity(const Real dt, const EOS &eos,
                                  const BoundaryCondition &bc,
                                  Fields<Real, Backend> &qq) {
    artdiff.characteristic_velocity_eval(qq, eos);

    // x direction
    artdiff.update(qq, qq_rslt, Direction::X, dt);
    qq.copy_from(qq_rslt);
    apply_boundary_condition(bc, qq);

    // y direction
    artdiff.update(qq, qq_rslt, Direction::Y, dt);
    qq.copy_from(qq_rslt);
    apply_boundary_condition(bc, qq);

    // z direction
    artdiff.update(qq, qq_rslt, Direction::Z, dt);
    qq.copy_from(qq_rslt);
    apply_boundary_condition(bc, qq);
  }

  /// @brief Calculate time spacing based on CFL condition
  template <typename EOS>
  Real cfl(const Fields<Real, Backend> &qq, const EOS &eos) {
    auto qq_v = qq.const_view();
    auto grid_v = grid.const_view();
    auto cs_v = cs.const_view();
    // device lambdas must not capture `this`
    const Real cfl_number_ = cfl_number;

    eos.sound_speed(Backend{}, qq_v, cs.view());
    const auto f = MISO_LAMBDA(int i, int j, int k) {
      // cs: sound speed, vv: fluid velocity, ca: Alfvén speed
      Real vv = util::sqrt(qq_v.vx(i, j, k) * qq_v.vx(i, j, k) +
                           qq_v.vy(i, j, k) * qq_v.vy(i, j, k) +
                           qq_v.vz(i, j, k) * qq_v.vz(i, j, k));
      Real ca = util::sqrt((qq_v.bx(i, j, k) * qq_v.bx(i, j, k) +
                            qq_v.by(i, j, k) * qq_v.by(i, j, k) +
                            qq_v.bz(i, j, k) * qq_v.bz(i, j, k)) /
                           qq_v.ro(i, j, k) * pii4<Real>);
      Real total_vel = cs_v(i, j, k) + vv + ca;
      Real dxyz = util::min3(grid_v.dx[i], grid_v.dy[j], grid_v.dz[k]);
      return cfl_number_ * dxyz / total_vel;
    };
    const auto op = MISO_LAMBDA(Real a, Real b) { return util::min2(a, b); };
    const Real dt_max = 1.e10;
    const Real dt = cfl_workspace.reduce(inner_range(), dt_max, f, op);

    Real dt_global;
    MPI_Allreduce(&dt, &dt_global, 1, mpi::data_type<Real>(), MPI_MIN,
                  mpi::comm());
    return dt_global;
  }

  /// @brief Set parameters for divergence B cleaning
  void divb_parameters_set(const Real dt) {
    ch_divb = 0.8 * cfl_number * grid.min_dxyz / dt;
    ch_divb_square = ch_divb * ch_divb;
    tau_divb = 2.0 * dt;
  }

  /// @brief Update MHD equations by one time step
  template <typename EOS, typename BoundaryCondition, typename Source>
  void update(Real dt, const EOS &eos, const BoundaryCondition &bc,
              const Source &src, Fields<Real, Backend> &qq) {
    divb_parameters_set(dt);
    runge_kutta_4step(dt, eos, bc, src, qq);
    apply_artificial_viscosity(dt, eos, bc, qq);
  }

  // Prohibit copy and move
  Integrator(const Integrator &) = delete;
  Integrator &operator=(const Integrator &) = delete;
  Integrator(Integrator &&) = delete;
  Integrator &operator=(Integrator &&) = delete;
};

}  // namespace mhd
}  // namespace miso
