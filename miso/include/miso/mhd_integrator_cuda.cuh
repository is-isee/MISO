#pragma once

#include "array3d.hpp"
#include "constants.hpp"
#include "cuda_compat.hpp"
#include "cuda_util.cuh"
#include "env.hpp"
#include "execution.hpp"
#include "mhd_artificial_viscosity_cuda.cuh"
#include "mhd_fields.hpp"
#include "mhd_halo_exchange.hpp"

namespace miso {
namespace mhd {

template <typename Real, typename EOS>
__global__ void pr_bb_ht_vb_kernel(FieldsView<const Real> qq_argm,
                                   Array3DView<Real> pr, Array3DView<Real> bb,
                                   Array3DView<Real> ht, Array3DView<Real> vb,
                                   GridView<const Real> grid, EOS eos) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < 0 || i >= grid.i_total || j < 0 || j >= grid.j_total || k < 0 ||
      k >= grid.k_total)
    return;

  // gas pressure
  pr(i, j, k) = eos.roeitopr(qq_argm.ro(i, j, k), qq_argm.ei(i, j, k));
  // squared magnetic strength
  bb(i, j, k) = qq_argm.bx(i, j, k) * qq_argm.bx(i, j, k) +
                qq_argm.by(i, j, k) * qq_argm.by(i, j, k) +
                qq_argm.bz(i, j, k) * qq_argm.bz(i, j, k);

  // enthalpy + 2*magnetic energy + kinetic energy
  // clang-format off
  ht(i, j, k) =
      +qq_argm.ro(i, j, k) * qq_argm.ei(i, j, k) +
      pr(i, j, k) + bb(i, j, k) * pii4<Real> +
      0.5 * qq_argm.ro(i, j, k) *
          (+ qq_argm.vx(i, j, k) * qq_argm.vx(i, j, k)
           + qq_argm.vy(i, j, k) * qq_argm.vy(i, j, k)
           + qq_argm.vz(i, j, k) * qq_argm.vz(i, j, k));
  // v dot b
  vb(i, j, k) =
      + qq_argm.vx(i, j, k) * qq_argm.bx(i, j, k)
      + qq_argm.vy(i, j, k) * qq_argm.by(i, j, k)
      + qq_argm.vz(i, j, k) * qq_argm.bz(i, j, k);
  // clang-format on
}

template <typename Real>
__device__ inline bool compute_index_within_margin(int &i, int &j, int &k,
                                                   GridView<const Real> grid) {
  i = blockIdx.x * blockDim.x + threadIdx.x + grid.i_margin;
  j = blockIdx.y * blockDim.y + threadIdx.y + grid.j_margin;
  k = blockIdx.z * blockDim.z + threadIdx.z + grid.k_margin;

  return !(i < grid.i_margin || i >= grid.i_total - grid.i_margin ||
           j < grid.j_margin || j >= grid.j_total - grid.j_margin ||
           k < grid.k_margin || k >= grid.k_total - grid.k_margin);
}

template <typename Real>
__global__ void
update_ro_kernel(FieldsView<const Real> qq_orgn, FieldsView<const Real> qq_argm,
                 FieldsView<Real> qq_rslt, GridView<const Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // equation of continuity
  // clang-format off
  qq_rslt.ro(i, j, k) = qq_orgn.ro(i, j, k) + dt * (
      - space_centered_4th(qq_argm.ro, qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0)
      - space_centered_4th(qq_argm.ro, qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0)
      - space_centered_4th(qq_argm.ro, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks));
  // clang-format on
}

template <typename Real, typename Source>
__global__ void
update_vx_kernel(FieldsView<const Real> qq_orgn, FieldsView<const Real> qq_argm,
                 FieldsView<Real> qq_rslt, Array3DView<const Real> pr,
                 Array3DView<const Real> bb, Source src,
                 GridView<const Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // clang-format off
  // x equation of motion
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
            + src.vx(qq_argm, i, j, k)
            )) / qq_rslt.ro(i, j, k);
  // clang-format on
}

template <typename Real, typename Source>
__global__ void
update_vy_kernel(FieldsView<const Real> qq_orgn, FieldsView<const Real> qq_argm,
                 FieldsView<Real> qq_rslt, Array3DView<const Real> pr,
                 Array3DView<const Real> bb, Source src,
                 GridView<const Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // y equation of motion
  qq_rslt.vy(i, j, k) =
      (qq_orgn.ro(i, j, k) * qq_orgn.vy(i, j, k) +
       // clang-format off
       dt *
           (- space_centered_4th(qq_argm.ro, qq_argm.vy, qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0)
            - space_centered_4th(qq_argm.ro, qq_argm.vy, qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0)
            - space_centered_4th(qq_argm.ro, qq_argm.vy, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks)
            - space_centered_4th(pr, grid.dyi[j], i, j, k, 0, grid.js, 0)
            - space_centered_4th(bb, grid.dyi[j], i, j, k, 0, grid.js, 0) * pii8<Real>
            + pii4<Real> * (+ space_centered_4th(qq_argm.by, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0)
                            + space_centered_4th(qq_argm.by, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0)
                            + space_centered_4th(qq_argm.by, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks))
            + src.vy(qq_argm, i, j, k)
            )) / qq_rslt.ro(i, j, k);
  // clang-format on
}

template <typename Real, typename Source>
__global__ void
update_vz_kernel(FieldsView<const Real> qq_orgn, FieldsView<const Real> qq_argm,
                 FieldsView<Real> qq_rslt, Array3DView<const Real> pr,
                 Array3DView<const Real> bb, Source src,
                 GridView<const Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // z equation of motion
  // clang-format off
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
             + src.vz(qq_argm, i, j, k)
          )) / qq_rslt.ro(i, j, k);
  // clang-format on
}

template <typename Real>
__global__ void
update_bx_kernel(FieldsView<const Real> qq_orgn, FieldsView<const Real> qq_argm,
                 FieldsView<Real> qq_rslt, GridView<const Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // x magnetic induction
  // clang-format off
  qq_rslt.bx(i, j, k) =
      qq_orgn.bx(i, j, k) +
      dt * (- space_centered_4th(qq_argm.vy, qq_argm.bx, grid.dyi[j], i, j, k, 0, grid.js, 0)
            - space_centered_4th(qq_argm.vz, qq_argm.bx, grid.dzi[k], i, j, k, 0, 0, grid.ks)
            + space_centered_4th(qq_argm.vx, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0)
            + space_centered_4th(qq_argm.vx, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks)
            - space_centered_4th(qq_argm.ph, grid.dxi[i], i, j, k, grid.is, 0, 0));
  // clang-format on
}

template <typename Real>
__global__ void
update_by_kernel(FieldsView<const Real> qq_orgn, FieldsView<const Real> qq_argm,
                 FieldsView<Real> qq_rslt, GridView<const Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // y magnetic induction
  // clang-format off
  qq_rslt.by(i, j, k) =
      qq_orgn.by(i, j, k) +
      dt * (- space_centered_4th(qq_argm.vx, qq_argm.by, grid.dxi[i], i, j, k, grid.is, 0, 0)
            - space_centered_4th(qq_argm.vz, qq_argm.by, grid.dzi[k], i, j, k, 0, 0, grid.ks)
            + space_centered_4th(qq_argm.vy, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0)
            + space_centered_4th(qq_argm.vy, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks)
            - space_centered_4th(qq_argm.ph, grid.dyi[j], i, j, k, 0, grid.js, 0));
  // clang-format on
}

template <typename Real>
__global__ void
update_bz_kernel(FieldsView<const Real> qq_orgn, FieldsView<const Real> qq_argm,
                 FieldsView<Real> qq_rslt, GridView<const Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // z magnetic induction
  // clang-format off
  qq_rslt.bz(i, j, k) =
      qq_orgn.bz(i, j, k) +
      dt * (- space_centered_4th(qq_argm.vx, qq_argm.bz, grid.dxi[i], i, j, k, grid.is, 0, 0)
            - space_centered_4th(qq_argm.vy, qq_argm.bz, grid.dyi[j], i, j, k, 0, grid.js, 0)
            + space_centered_4th(qq_argm.vz, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0)
            + space_centered_4th(qq_argm.vz, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0)
            - space_centered_4th(qq_argm.ph, grid.dzi[k], i, j, k, 0, 0, grid.ks));
  // clang-format on
}

template <typename Real>
__global__ void
update_ph_kernel(FieldsView<const Real> qq_orgn, FieldsView<const Real> qq_argm,
                 FieldsView<Real> qq_rslt, Real ch_divb_square, Real tau_divb,
                 GridView<const Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // div B factor
  // clang-format off
  qq_rslt.ph(i, j, k) =
      (qq_orgn.ph(i, j, k) +
       dt * ch_divb_square *
           (- space_centered_4th(qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0)
            - space_centered_4th(qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0)
            - space_centered_4th(qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks))) *
      util::exp(-dt / tau_divb);
  // clang-format on
}

template <typename Real, typename Source>
__global__ void
update_ei_kernel(FieldsView<const Real> qq_orgn, FieldsView<const Real> qq_argm,
                 FieldsView<Real> qq_rslt, Array3DView<const Real> pr,
                 Array3DView<const Real> bb, Array3DView<const Real> ht,
                 Array3DView<const Real> vb, Source src,
                 GridView<const Real> grid, Real dt) {
  int i, j, k;
  if (!compute_index_within_margin(i, j, k, grid))
    return;

  // Et: total energy per unit volume
  // ei: is the internal energy per unit mass
  // clang-format off
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
       + src.ei(qq_argm, i, j, k)
      ) / qq_rslt.ro(i, j, k);
  // clang-format on
}

template <typename Real> struct Integrator<Real, backend::CUDA> {
  /// @brief CUDA kernel shape
  cuda::KernelShape3D &cu_shape;

  /// @brief Spatial grid
  Grid<Real, backend::CUDA> &grid;
  /// @brief Workspace
  Fields<Real, backend::CUDA> qq_argm, qq_rslt;

  /// @brief Halo exchanger
  HaloExchanger<Real, backend::CUDA> halo_exchanger;
  /// @brief Artificial viscosity for MHD equations
  impl_cuda::ArtificialViscosity<Real> artdiff;

  /// @brief Workspace for timestep calculation
  ReduceHelper<Real> cfl_helper;

  /// @brief gas pressure
  Array3D<Real, backend::CUDA> pr;
  /// @brief magnetic field strength bx*bx + by*by + bz*bz
  Array3D<Real, backend::CUDA> bb;
  /// @brief enthalpy + 2*magnetic energy + kinetic energy
  Array3D<Real, backend::CUDA> ht;
  /// @brief inner product of velocity and magnetic field vx*bx + vy*by + vz*bz
  Array3D<Real, backend::CUDA> vb;

  /// @brief CFL number
  Real cfl_number;
  /// @brief propagation speed fo divergence B
  Real ch_divb;
  /// @brief square of ch_divb;
  Real ch_divb_square;
  /// @brief damping time scape for divergence B
  Real tau_divb;

  Integrator(Config &config, Grid<Real, backend::CUDA> &grid,
             ExecContext<Real, backend::CUDA> &exec_ctx)
      : cu_shape(exec_ctx.cu_shape), grid(grid), qq_argm(grid), qq_rslt(grid),
        halo_exchanger(grid, exec_ctx), artdiff(config, grid, exec_ctx.cu_shape),
        pr(grid.i_total, grid.j_total, grid.k_total),
        bb(grid.i_total, grid.j_total, grid.k_total),
        ht(grid.i_total, grid.j_total, grid.k_total),
        vb(grid.i_total, grid.j_total, grid.k_total) {
    cfl_number = config["mhd"]["cfl_number"].as<Real>();
  }

  /// @brief Update MHD equations using 4th order space-centered scheme
  template <typename EOS, typename Source>
  void update_sc4(const Real dt, const EOS &eos, const Source &src,
                  const Fields<Real, backend::CUDA> &qq_orgn,
                  const Fields<Real, backend::CUDA> &qq_argm,
                  Fields<Real, backend::CUDA> &qq_rslt) {
    const auto &cgrid = grid.const_view();

    pr_bb_ht_vb_kernel<Real, EOS><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_argm.view(), pr.view(), bb.view(), ht.view(), vb.view(), cgrid, eos);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_ro_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), cgrid, dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_vx_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), pr.const_view(),
        bb.const_view(), src, cgrid, dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_vy_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), pr.const_view(),
        bb.const_view(), src, cgrid, dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_vz_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), pr.const_view(),
        bb.const_view(), src, cgrid, dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_bx_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), cgrid, dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_by_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), cgrid, dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_bz_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), cgrid, dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_ph_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), ch_divb_square, tau_divb,
        cgrid, dt);
    MISO_CUDA_CHECK(cudaGetLastError());

    update_ei_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
        qq_orgn.view(), qq_argm.view(), qq_rslt.view(), pr.const_view(),
        bb.const_view(), ht.const_view(), vb.const_view(), src, cgrid, dt);
    MISO_CUDA_CHECK(cudaGetLastError());
  }

  /// @brief Apply boundary condition and halo exchange
  template <typename BoundaryCondition>
  void apply_boundary_condition(const BoundaryCondition &bc,
                                Fields<Real, backend::CUDA> &qq) {
    bc.apply(qq.view(), grid.const_view());
    MISO_CUDA_CHECK(cudaDeviceSynchronize());  // May not be necessary
    halo_exchanger.apply(qq);
  }

  /// @brief Runge-Kutta 4th order time integration step
  template <typename EOS, typename BoundaryCondition, typename Source>
  void runge_kutta_4step(const Real dt, const EOS &eos,
                         const BoundaryCondition &bc, const Source &src,
                         Fields<Real, backend::CUDA> &qq) {
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
                                  Fields<Real, backend::CUDA> &qq) {
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
  Real cfl(const Fields<Real, backend::CUDA> &qq, const EOS &eos) {
    auto qq_v = qq.const_view();
    auto grid_v = grid.const_view();

    Range3D range{{grid.i_margin, grid.i_total - grid.i_margin},
                  {grid.j_margin, grid.j_total - grid.j_margin},
                  {grid.k_margin, grid.k_total - grid.k_margin}};
    const auto f = MISO_LAMBDA(int i, int j, int k) {
      Real cs = eos.roeitocs(qq_v.ro(i, j, k), qq_v.ei(i, j, k));
      Real vv = util::sqrt(qq_v.vx(i, j, k) * qq_v.vx(i, j, k) +
                           qq_v.vy(i, j, k) * qq_v.vy(i, j, k) +
                           qq_v.vz(i, j, k) * qq_v.vz(i, j, k));
      Real ca = util::sqrt((qq_v.bx(i, j, k) * qq_v.bx(i, j, k) +
                            qq_v.by(i, j, k) * qq_v.by(i, j, k) +
                            qq_v.bz(i, j, k) * qq_v.bz(i, j, k)) /
                           qq_v.ro(i, j, k) * pii4<Real>);
      Real total_vel = (cs + vv + ca);
      Real dxyz = util::min3(grid_v.dx[i], grid_v.dy[j], grid_v.dz[k]);
      return cfl_number * dxyz / total_vel;
    };
    const auto op = MISO_LAMBDA(Real a, Real b) { return util::min2(a, b); };
    const Real dt_max = 1.e10;
    const auto dt = reduce(backend::CUDA{}, range, dt_max, f, op, cfl_helper);

    Real dt_g;
    MPI_Allreduce(&dt, &dt_g, 1, mpi::data_type<Real>(), MPI_MIN, mpi::comm());
    return dt_g;
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
              const Source &src, Fields<Real, backend::CUDA> &qq) {
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
