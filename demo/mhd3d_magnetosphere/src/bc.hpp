#pragma once
#include "boundary_condition_base.hpp"
#include "boundary_condition_core.hpp"
#include "initial_condition.hpp"
#include "model.hpp"
#include "standard_boundary_condition.hpp"
#include "utility.hpp"
#include <memory>
#ifdef USE_CUDA
#include "array3d_gpu.cuh"
#include "boundary_condition_core_gpu.cuh"
#include "cuda_manager.cuh"
#else
#include "boundary_condition_core_cpu.hpp"
#endif

template <typename Real, typename MHDCoreType, typename GridType,
          typename Array3DType>
HOST_DEVICE inline void
geo_boundary_condition_core(MHDCoreType &qq, const MHDCoreType &qq_init,
                            const GridType &grid, int i, int j, int k,
                            const Array3DType &f_sphere, EOS<Real> eos,
                            Real ro_floor, Real pr_floor) {
  Real rr =
      sqrt(grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] + grid.z[k] * grid.z[k]);

  // clang-format off
  // Helper lambda for linear interpolation
  auto lerp = [](Real a, Real b, Real f) -> Real {
    return a * f + b * (1.0 - f);
  };
  int idx = grid.idx(i, j, k);
  Real f = f_sphere[idx];
  qq.ro[idx] = lerp(qq.ro[idx], qq_init.ro[idx], f);
  qq.vx[idx] = lerp(qq.vx[idx], qq_init.vx[idx], f);
  qq.vy[idx] = lerp(qq.vy[idx], qq_init.vy[idx], f);
  qq.vz[idx] = lerp(qq.vz[idx], qq_init.vz[idx], f);
  qq.bx[idx] = lerp(qq.bx[idx], qq_init.bx[idx], f);
  qq.by[idx] = lerp(qq.by[idx], qq_init.by[idx], f);
  qq.bz[idx] = lerp(qq.bz[idx], qq_init.bz[idx], f);
  qq.ei[idx] = lerp(qq.ei[idx], qq_init.ei[idx], f);
  qq.ph[idx] = lerp(qq.ph[idx], qq_init.ph[idx], f);
  // clang-format on

  qq.ro[grid.idx(i, j, k)] = util::max2<Real>(qq.ro[grid.idx(i, j, k)], ro_floor);
  qq.ei[grid.idx(i, j, k)] =
      util::max2<Real>(qq.ei[grid.idx(i, j, k)],
                       pr_floor / (eos.gm - 1.0) / qq.ro[grid.idx(i, j, k)]);
}

template <typename Real, typename MHDCoreType, typename GridType>
HOST_DEVICE inline void
solar_wind_boundary_condition_core(MHDCoreType &qq, const GridType &grid, int i,
                                   int j, int k, EOS<Real> eos, Real ro_sw,
                                   Real pr_sw, Real vx_sw, Real bz_imf) {
  int i_ghst, i_trgt;
  bnd::symmetric_index<Real>(i, grid.i_total, grid.i_margin, i_ghst, i_trgt,
                             bnd::Side::INNER);
  qq.ro[grid.idx(i_ghst, j, k)] = ro_sw;
  qq.vx[grid.idx(i_ghst, j, k)] = vx_sw;
  qq.ei[grid.idx(i_ghst, j, k)] =
      pr_sw / (eos.gm - 1.0) / qq.ro[grid.idx(i_ghst, j, k)];
  qq.bz[grid.idx(i_ghst, j, k)] = bz_imf;
}

#ifdef USE_CUDA
template <typename Real>
__global__ inline void geo_boundary_condition_kernel(
    MHDCoreDevice<Real> qq, const MHDCoreDevice<Real> qq_init,
    const Array3DDevice<Real> f_sphere, const GridDevice<Real> grid,
    const EOS<Real> eos, Real ro_floor, Real pr_floor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < grid.i_total && j < grid.j_total && k < grid.k_total) {
    geo_boundary_condition_core<Real, MHDCoreDevice<Real>, GridDevice<Real>,
                                Array3DDevice<Real>>(
        qq, qq_init, grid, i, j, k, f_sphere, eos, ro_floor, pr_floor);
  }
}

template <typename Real>
__global__ void solar_wind_boundary_condition_kernel(
    MHDCoreDevice<Real> qq, const GridDevice<Real> grid, const EOS<Real> eos,
    Real ro_sw, Real pr_sw, Real vx_sw, Real bz_imf, Real rra) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int i0_, i1_, j0_, j1_, k0_, k1_;
  bnd::range_set<Real>(i0_, i1_, j0_, j1_, k0_, k1_, bnd::Direction::X, grid);
  if (i >= i0_ && i < i1_ && j >= j0_ && j < j1_ && k >= k0_ && k < k1_) {
    solar_wind_boundary_condition_core<Real, MHDCoreDevice<Real>,
                                       GridDevice<Real>>(
        qq, grid, i, j, k, eos, ro_sw, pr_sw, vx_sw, bz_imf);
  }
}

template <typename Real>
void apply_custom_boundary_condition_core(
    MHDCoreDevice<Real> qq, const MHDCoreDevice<Real> qq_init,
    const Array3DDevice<Real> f_sphere, const GridDevice<Real> grid,
    const EOS<Real> eos, Real ro_sw, Real pr_sw, Real vx_sw, Real bz_imf,
    Real rra, Real ro_floor, Real pr_floor, MHDCudaManager<Real> &cu_shape) {
  geo_boundary_condition_kernel<Real><<<cu_shape.grid_dim, cu_shape.block_dim>>>(
      qq, qq_init, f_sphere, grid, eos, ro_floor, pr_floor);
  solar_wind_boundary_condition_kernel<Real>
      <<<cu_shape.grid_dim, cu_shape.block_dim>>>(qq, grid, eos, ro_sw, pr_sw,
                                                  vx_sw, bz_imf, rra);
}

#else
template <typename Real>
void apply_custom_boundary_condition_core(
    MHDCore<Real> &qq, const MHDCore<Real> &qq_init,
    const Array3D<Real> &f_sphere, const Grid<Real> &grid, const EOS<Real> &eos,
    Real ro_sw, Real pr_sw, Real vx_sw, Real bz_imf, Real rra, Real ro_floor,
    Real pr_floor) {
  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      for (int k = 0; k < grid.k_total; ++k) {
        geo_boundary_condition_core<Real, MHDCore<Real>, Grid<Real>,
                                    Array3D<Real>>(
            qq, qq_init, grid, i, j, k, f_sphere, eos, ro_floor, pr_floor);
      }
    }
  }

  int i0_, i1_, j0_, j1_, k0_, k1_;
  bnd::range_set<Real, Grid<Real>>(i0_, i1_, j0_, j1_, k0_, k1_,
                                   bnd::Direction::X, grid);
  for (int i = i0_; i < i1_; ++i) {
    for (int j = j0_; j < j1_; ++j) {
      for (int k = k0_; k < k1_; ++k) {
        solar_wind_boundary_condition_core<Real, MHDCore<Real>, Grid<Real>>(
            qq, grid, i, j, k, eos, ro_sw, pr_sw, vx_sw, bz_imf);
      }
    }
  }
}
#endif

template <typename Real, typename MHDCoreType, typename GridType>
struct CustomBoundaryCondition
    : public BoundaryConditionBase<Real, MHDCoreType, GridType> {
  Config &config;
  Grid<Real> &grid;
  EOS<Real> &eos;
  MPIManager &mpi;
  MHDCore<Real> qq_init;
  Array3D<Real> f_sphere;

#ifdef USE_CUDA
  MHDCoreDevice<Real> qq_init_d;
  Array3DDevice<Real> f_sphere_d;
  GridDevice<Real> &grid_d;
  MHDCudaManager<Real> &cu_shape;
#endif

  std::unique_ptr<BoundaryConditionBase<Real, MHDCoreType, GridType>> bc_standard;
  Real rra, a0;
  Real ro_sw, pr_sw, vx_sw, bz_imf;
  Real ro_floor, pr_floor;

  CustomBoundaryCondition(Model<Real> &model)
      : config(model.config), grid(model.grid_local),
#ifdef USE_CUDA
        grid_d(model.grid_d), qq_init_d(grid),
        f_sphere_d(grid.i_total, grid.j_total, grid.k_total),
        cu_shape(model.cu_shape),
#endif
        eos(model.eos), mpi(model.mpi),
        qq_init(grid.i_total, grid.j_total, grid.k_total),
        f_sphere(grid.i_total, grid.j_total, grid.k_total) {

    rra = config.yaml_obj["geo_boundary"]["radius"].template as<Real>();
    a0 = config.yaml_obj["geo_boundary"]["a0"].template as<Real>();

    ro_sw = config.yaml_obj["solar_wind"]["ro_sw"].template as<Real>();
    pr_sw = config.yaml_obj["solar_wind"]["pr_sw"].template as<Real>();
    vx_sw = config.yaml_obj["solar_wind"]["vx_sw"].template as<Real>();
    bz_imf = config.yaml_obj["solar_wind"]["bz_imf"].template as<Real>();

    ro_floor = config.yaml_obj["floor"]["ro_floor"].template as<Real>();
    pr_floor = config.yaml_obj["floor"]["pr_floor"].template as<Real>();

    bc_standard =
        std::make_unique<StandardBoundaryCondition<Real, MHDCoreType, GridType>>(
            model);
    InitialCondition<Real> initial_condition(model);
    initial_condition.apply(qq_init);

    for (int i = 0; i < grid.i_total; ++i) {
      for (int j = 0; j < grid.j_total; ++j) {
        for (int k = 0; k < grid.k_total; ++k) {
          Real rr = std::sqrt(grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] +
                              grid.z[k] * grid.z[k]);
          Real hh;
          if (rr > rra) {
            hh = util::pow2(rr / rra) - 1.0;
          } else {
            hh = 0.0;
            grid.mask(i, j, k) = 0.0;  // inside the inner boundary is masked
          }
          f_sphere(i, j, k) = a0 * hh / (a0 * hh + 1.0);
        }
      }
    }

#ifdef USE_CUDA
    qq_init_d.copy_from_host(qq_init, model.cu_shape);
    f_sphere_d.copy_from_host(f_sphere);
#endif
  }

  inline void apply(MHDCoreType &qq) override {
    // apply standard boundary condition once
    bc_standard->apply(qq);
#ifdef USE_CUDA
    apply_custom_boundary_condition_core<Real>(qq, qq_init_d, f_sphere_d, grid_d,
                                               eos, ro_sw, pr_sw, vx_sw, bz_imf,
                                               rra, ro_floor, pr_floor, cu_shape);
#else
    apply_custom_boundary_condition_core<Real>(qq, qq_init, f_sphere, grid, eos,
                                               ro_sw, pr_sw, vx_sw, bz_imf, rra,
                                               ro_floor, pr_floor);
#endif
  }
};
