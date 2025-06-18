#pragma once

#include <cmath>
#include <algorithm>
#include <initializer_list>
#include <filesystem>
#include <mpi.h>
#include <cuda_runtime.h>

#include "model.hpp"
#include "array3d_gpu.cuh"
#include "grid_gpu.cuh"
#include "constants.hpp"
#include "standard_boundary_condition.hpp"
#include "custom_boundary_condition.hpp"
#include "artificial_viscosity_gpu.hpp"
#include "mpi_types.hpp"
#include "cuda_manager.cuh"
#include "time_gpu.cuh"
#include "utility.hpp"

template <typename Real>
__global__ void cfl_condition_kernel(
    MHDCoreDevice<Real> qq, GridDevice<Real> grid, Real* dt_mins, Real cfl_number, Real eos_gm
) {
    extern __shared__ Real dt_min_shared_in_block[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;


    Real dt = 1.e10;
    if (i >= grid.i_margin && i < grid.i_total - grid.i_margin && 
        j >= grid.j_margin && j < grid.j_total - grid.j_margin && 
        k >= grid.k_margin && k < grid.k_total - grid.k_margin) {
        Real cs = std::sqrt(eos_gm*(eos_gm-1.0)*qq.ei[grid.idx(i,j,k)]);
        Real vv = std::sqrt(
            + qq.vx[grid.idx(i,j,k)]*qq.vx[grid.idx(i,j,k)]
            + qq.vy[grid.idx(i,j,k)]*qq.vy[grid.idx(i,j,k)]
            + qq.vz[grid.idx(i,j,k)]*qq.vz[grid.idx(i,j,k)]
        );  
        Real ca = std::sqrt( (
            + qq.bx[grid.idx(i,j,k)]*qq.bx[grid.idx(i,j,k)]
            + qq.by[grid.idx(i,j,k)]*qq.by[grid.idx(i,j,k)]
            + qq.bz[grid.idx(i,j,k)]*qq.bz[grid.idx(i,j,k)]
        )/qq.ro[grid.idx(i,j,k)]*pii4<Real>);
        dt = cfl_number*util::min3(grid.dx[i], grid.dy[j], grid.dz[k])/(cs + vv + ca);
    }

    int thread_id = threadIdx.z * blockDim.y * blockDim.x +
                    threadIdx.y * blockDim.x +
                    threadIdx.x;
    dt_min_shared_in_block[thread_id] = dt;
    __syncthreads();

    for (int s = blockDim.x * blockDim.y * blockDim.z / 2; s > 0; s >>= 1) {
        if (thread_id < s) {
            dt_min_shared_in_block[thread_id] = util::fmin_safe(dt_min_shared_in_block[thread_id], dt_min_shared_in_block[thread_id + s]);
        }
        __syncthreads();
    }

    // ブロックのスレッド0が結果を書き出し
    if (thread_id == 0) {
        int block_id = blockIdx.z * gridDim.y * gridDim.x +
                       blockIdx.y * gridDim.x +
                       blockIdx.x;
        dt_mins[block_id] = dt_min_shared_in_block[0];
    }
}


template <typename Real>
__device__ inline Real space_centered_4th(const Real* qq1, Real dxyzi, int i, int j, int k, int is, int js, int ks, const GridDevice<Real>& grid) {
    return (
        -     qq1[grid.idx(i + 2*is, j + 2*js, k + 2*ks)]
        + 8.0*qq1[grid.idx(i +   is, j +   js, k +   ks)]
        - 8.0*qq1[grid.idx(i -   is, j -   js, k -   ks)]
        +     qq1[grid.idx(i - 2*is, j - 2*js, k - 2*ks)]
    )*inv12<Real>*dxyzi;
};

template <typename Real>
__device__ inline Real space_centered_4th(const Real* qq1, const Real* qq2, Real dxyzi, int i, int j, int k, int is, int js, int ks, const GridDevice<Real>& grid) {
    return (
        -     qq1[grid.idx(i + 2*is, j + 2*js, k + 2*ks)]*qq2[grid.idx(i + 2*is, j + 2*js, k + 2*ks)]
        + 8.0*qq1[grid.idx(i +   is, j +   js, k +   ks)]*qq2[grid.idx(i +   is, j +   js, k +   ks)]
        - 8.0*qq1[grid.idx(i -   is, j -   js, k -   ks)]*qq2[grid.idx(i -   is, j -   js, k -   ks)]
        +     qq1[grid.idx(i - 2*is, j - 2*js, k - 2*ks)]*qq2[grid.idx(i - 2*is, j - 2*js, k - 2*ks)]
    )*inv12<Real>*dxyzi;
};

template <typename Real>
__device__ inline Real space_centered_4th(const Real* qq1, const Real* qq2, const Real* qq3, Real dxyzi, int i, int j, int k, int is, int js, int ks, const GridDevice<Real>& grid) {
    return (
        -     qq1[grid.idx(i + 2*is, j + 2*js, k + 2*ks)]*qq2[grid.idx(i + 2*is, j + 2*js, k + 2*ks)]*qq3[grid.idx(i + 2*is, j + 2*js, k + 2*ks)]
        + 8.0*qq1[grid.idx(i +   is, j +   js, k +   ks)]*qq2[grid.idx(i +   is, j +   js, k +   ks)]*qq3[grid.idx(i +   is, j +   js, k +   ks)]
        - 8.0*qq1[grid.idx(i -   is, j -   js, k -   ks)]*qq2[grid.idx(i -   is, j -   js, k -   ks)]*qq3[grid.idx(i -   is, j -   js, k -   ks)]
        +     qq1[grid.idx(i - 2*is, j - 2*js, k - 2*ks)]*qq2[grid.idx(i - 2*is, j - 2*js, k - 2*ks)]*qq3[grid.idx(i - 2*is, j - 2*js, k - 2*ks)]
    )*inv12<Real>*dxyzi;
};

template <typename Real>
__global__ void pr_bb_ht_vb_kernel(
    MHDCoreDevice<Real> qq_argm,
    Array3DDevice<Real> pr, 
    Array3DDevice<Real> bb,
    Array3DDevice<Real> ht,
    Array3DDevice<Real> vb,
    GridDevice<Real> grid,
    Real eos_gm )
    {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < 0 || i >= grid.i_total ||
        j < 0 || j >= grid.j_total ||
        k < 0 || k >= grid.k_total )
        return;

    // gas pressure
    pr[grid.idx(i, j, k)] = qq_argm.ro[grid.idx(i,j,k)]*qq_argm.ei[grid.idx(i,j,k)]*(eos_gm - 1.0);
    // squared magnetic strength
    bb[grid.idx(i,j,k)] = 
        qq_argm.bx[grid.idx(i,j,k)]*qq_argm.bx[grid.idx(i,j,k)]
      + qq_argm.by[grid.idx(i,j,k)]*qq_argm.by[grid.idx(i,j,k)]
      + qq_argm.bz[grid.idx(i,j,k)]*qq_argm.bz[grid.idx(i,j,k)];

    // // enthalpy + 2*magnetic energy + kinetic energy      
    ht[grid.idx(i, j, k)] = 
        + qq_argm.ro[grid.idx(i,j,k)]*qq_argm.ei[grid.idx(i,j,k)] + pr[grid.idx(i,j,k)]
        + bb[grid.idx(i,j,k)]*pii4<Real>
        + 0.5*qq_argm.ro[grid.idx(i,j,k)]*(
            + qq_argm.vx[grid.idx(i,j,k)]*qq_argm.vx[grid.idx(i,j,k)]
            + qq_argm.vy[grid.idx(i,j,k)]*qq_argm.vy[grid.idx(i,j,k)]
            + qq_argm.vz[grid.idx(i,j,k)]*qq_argm.vz[grid.idx(i,j,k)]
        );
    // v dot b
    vb[grid.idx(i, j, k)] = 
        + qq_argm.vx[grid.idx(i,j,k)]*qq_argm.bx[grid.idx(i,j,k)]
        + qq_argm.vy[grid.idx(i,j,k)]*qq_argm.by[grid.idx(i,j,k)]
        + qq_argm.vz[grid.idx(i,j,k)]*qq_argm.bz[grid.idx(i,j,k)];
}

template <typename Real>
__device__ inline bool compute_index_within_margin(
    int &i, int &j, int &k,
    const GridDevice<Real>& grid)
{
    i = blockIdx.x * blockDim.x + threadIdx.x + grid.i_margin;
    j = blockIdx.y * blockDim.y + threadIdx.y + grid.j_margin;
    k = blockIdx.z * blockDim.z + threadIdx.z + grid.k_margin;

    return !(i < grid.i_margin || i >= grid.i_total - grid.i_margin ||
             j < grid.j_margin || j >= grid.j_total - grid.j_margin ||
             k < grid.k_margin || k >= grid.k_total - grid.k_margin);
}

template <typename Real>
__global__ void update_ro_kernel(
    MHDCoreDevice<Real> qq_orgn, MHDCoreDevice<Real> qq_argm, MHDCoreDevice<Real> qq_rslt,
    GridDevice<Real> grid, 
    Real dt
)
{
    int i, j, k;
    if(!compute_index_within_margin(i, j, k, grid)) return;

    // equation of continuity
    qq_rslt.ro[grid.idx(i, j, k)] = qq_orgn.ro[grid.idx(i, j, k)] + dt * (
        - space_centered_4th(qq_argm.ro, qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
        - space_centered_4th(qq_argm.ro, qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
        - space_centered_4th(qq_argm.ro, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
    );
}

template <typename Real>
__global__ void update_vx_kernel(
    MHDCoreDevice<Real> qq_orgn, MHDCoreDevice<Real> qq_argm, MHDCoreDevice<Real> qq_rslt,
    Array3DDevice<Real> pr, Array3DDevice<Real> bb,
    GridDevice<Real> grid, 
    Real dt
)
{
    int i, j, k;
    if(!compute_index_within_margin(i, j, k, grid)) return;

    // x equation of motion
    qq_rslt.vx[grid.idx(i, j, k)] = (
        qq_orgn.ro[grid.idx(i, j, k)] * qq_orgn.vx[grid.idx(i, j, k)] + dt * (
        -space_centered_4th(qq_argm.ro, qq_argm.vx, qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
        -space_centered_4th(qq_argm.ro, qq_argm.vx, qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
        -space_centered_4th(qq_argm.ro, qq_argm.vx, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
        -space_centered_4th(pr.data(), grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
        -space_centered_4th(bb.data(), grid.dxi[i], i, j, k, grid.is, 0, 0, grid)*pii8<Real>
        +pii4<Real>*(
            +space_centered_4th(qq_argm.bx, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
            +space_centered_4th(qq_argm.bx, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
            +space_centered_4th(qq_argm.bx, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
            )
        )
    )/qq_rslt.ro[grid.idx(i, j, k)];
}

template <typename Real>
__global__ void update_vy_kernel(
    MHDCoreDevice<Real> qq_orgn, MHDCoreDevice<Real> qq_argm, MHDCoreDevice<Real> qq_rslt,
    Array3DDevice<Real> pr, Array3DDevice<Real> bb,
    GridDevice<Real> grid, 
    Real dt
)
{
    int i, j, k;
    if(!compute_index_within_margin(i, j, k, grid)) return;

    // y equation of motion
    qq_rslt.vy[grid.idx(i, j, k)] = (
        qq_orgn.ro[grid.idx(i, j, k)] * qq_orgn.vy[grid.idx(i, j, k)] + dt * (
        -space_centered_4th(qq_argm.ro, qq_argm.vy, qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
        -space_centered_4th(qq_argm.ro, qq_argm.vy, qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
        -space_centered_4th(qq_argm.ro, qq_argm.vy, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
        -space_centered_4th(pr.data(), grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
        -space_centered_4th(bb.data(), grid.dyi[j], i, j, k, 0, grid.js, 0, grid)*pii8<Real>
        +pii4<Real>*(
            +space_centered_4th(qq_argm.by, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
            +space_centered_4th(qq_argm.by, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
            +space_centered_4th(qq_argm.by, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
            )
        )
    )/qq_rslt.ro[grid.idx(i, j, k)];
}

template <typename Real>
__global__ void update_vz_kernel(
    MHDCoreDevice<Real> qq_orgn, MHDCoreDevice<Real> qq_argm, MHDCoreDevice<Real> qq_rslt,
    Array3DDevice<Real> pr, Array3DDevice<Real> bb,
    GridDevice<Real> grid, 
    Real dt
)
{
    int i, j, k;
    if(!compute_index_within_margin(i, j, k, grid)) return;

    // z equation of motion
    qq_rslt.vz[grid.idx(i, j, k)] = (
        qq_orgn.ro[grid.idx(i, j, k)] * qq_orgn.vz[grid.idx(i, j, k)] + dt * (
        -space_centered_4th(qq_argm.ro, qq_argm.vz, qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
        -space_centered_4th(qq_argm.ro, qq_argm.vz, qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
        -space_centered_4th(qq_argm.ro, qq_argm.vz, qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
        -space_centered_4th(pr.data(), grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
        -space_centered_4th(bb.data(), grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)*pii8<Real>
        +pii4<Real>*(
            +space_centered_4th(qq_argm.bz, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
            +space_centered_4th(qq_argm.bz, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
            +space_centered_4th(qq_argm.bz, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
            )
        )
    )/qq_rslt.ro[grid.idx(i, j, k)];
}

template <typename Real>
__global__ void update_bx_kernel(
    MHDCoreDevice<Real> qq_orgn, MHDCoreDevice<Real> qq_argm, MHDCoreDevice<Real> qq_rslt,
    GridDevice<Real> grid, 
    Real dt
)
{
    int i, j, k;
    if(!compute_index_within_margin(i, j, k, grid)) return;

    // x magnetic induction
    qq_rslt.bx[grid.idx(i, j, k)] = qq_orgn.bx[grid.idx(i, j, k)] + dt * (
        -space_centered_4th(qq_argm.vy, qq_argm.bx, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
        -space_centered_4th(qq_argm.vz, qq_argm.bx, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
        +space_centered_4th(qq_argm.vx, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
        +space_centered_4th(qq_argm.vx, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
        -space_centered_4th(qq_argm.ph, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
    );
}

template <typename Real>
__global__ void update_by_kernel(
    MHDCoreDevice<Real> qq_orgn, MHDCoreDevice<Real> qq_argm, MHDCoreDevice<Real> qq_rslt,
    GridDevice<Real> grid, 
    Real dt
)
{
    int i, j, k;
    if(!compute_index_within_margin(i, j, k, grid)) return;

    // y magnetic induction
    qq_rslt.by[grid.idx(i, j, k)] = qq_orgn.by[grid.idx(i, j, k)] + dt * (
        -space_centered_4th(qq_argm.vx, qq_argm.by, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
        -space_centered_4th(qq_argm.vz, qq_argm.by, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
        +space_centered_4th(qq_argm.vy, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
        +space_centered_4th(qq_argm.vy, qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
        -space_centered_4th(qq_argm.ph, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
    );
}

template <typename Real>
__global__ void update_bz_kernel(
    MHDCoreDevice<Real> qq_orgn, MHDCoreDevice<Real> qq_argm, MHDCoreDevice<Real> qq_rslt,
    GridDevice<Real> grid, 
    Real dt
)
{
    int i, j, k;
    if(!compute_index_within_margin(i, j, k, grid)) return;

    // z magnetic induction
    qq_rslt.bz[grid.idx(i, j, k)] = qq_orgn.bz[grid.idx(i, j, k)] + dt * (
        -space_centered_4th(qq_argm.vx, qq_argm.bz, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
        -space_centered_4th(qq_argm.vy, qq_argm.bz, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
        +space_centered_4th(qq_argm.vz, qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
        +space_centered_4th(qq_argm.vz, qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
        -space_centered_4th(qq_argm.ph, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
    );
}

template <typename Real>
__global__ void update_ph_kernel(
    MHDCoreDevice<Real> qq_orgn, MHDCoreDevice<Real> qq_argm, MHDCoreDevice<Real> qq_rslt,
    Real ch_divb_square, Real tau_divb,
    GridDevice<Real> grid, 
    Real dt
)
{
    int i, j, k;
    if(!compute_index_within_margin(i, j, k, grid)) return;

    // div B factor
    qq_rslt.ph[grid.idx(i, j, k)] = (
        qq_orgn.ph[grid.idx(i, j, k)] + dt * ch_divb_square*(
        -space_centered_4th(qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
        -space_centered_4th(qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
        -space_centered_4th(qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
        )
    )*std::exp(-dt/tau_divb);
}

template <typename Real>
__global__ void update_ei_kernel(
    MHDCoreDevice<Real> qq_orgn, MHDCoreDevice<Real> qq_argm, MHDCoreDevice<Real> qq_rslt,
    Array3DDevice<Real> pr, Array3DDevice<Real> bb, Array3DDevice<Real> ht, Array3DDevice<Real> vb,
    GridDevice<Real> grid, 
    Real dt
)
{
    int i, j, k;
    if(!compute_index_within_margin(i, j, k, grid)) return;

    // Et: total energy per unit volume
    // ei: is the internal energy per unit mass 
    const Real Et = 
        + qq_orgn.ro[grid.idx(i, j, k)]*qq_orgn.ei[grid.idx(i, j, k)]
        + 0.5*qq_orgn.ro[grid.idx(i, j, k)]*(
            + qq_orgn.vx[grid.idx(i, j, k)]*qq_orgn.vx[grid.idx(i, j, k)]
            + qq_orgn.vy[grid.idx(i, j, k)]*qq_orgn.vy[grid.idx(i, j, k)]
            + qq_orgn.vz[grid.idx(i, j, k)]*qq_orgn.vz[grid.idx(i, j, k)]
        )
        + pii8<Real>*(
            + qq_orgn.bx[grid.idx(i, j, k)]*qq_orgn.bx[grid.idx(i, j, k)]
            + qq_orgn.by[grid.idx(i, j, k)]*qq_orgn.by[grid.idx(i, j, k)]
            + qq_orgn.bz[grid.idx(i, j, k)]*qq_orgn.bz[grid.idx(i, j, k)]
        );

    qq_rslt.ei[grid.idx(i, j, k)] = ( Et + dt * (
        -space_centered_4th(ht.data(), qq_argm.vx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
        -space_centered_4th(ht.data(), qq_argm.vy, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
        -space_centered_4th(ht.data(), qq_argm.vz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)
        + pii4<Real>* (
            +space_centered_4th(vb.data(), qq_argm.bx, grid.dxi[i], i, j, k, grid.is, 0, 0, grid)
            +space_centered_4th(vb.data(), qq_argm.by, grid.dyi[j], i, j, k, 0, grid.js, 0, grid)
            +space_centered_4th(vb.data(), qq_argm.bz, grid.dzi[k], i, j, k, 0, 0, grid.ks, grid)  
            )
        )   
        -0.5*qq_rslt.ro[grid.idx(i, j, k)]*(
            + qq_rslt.vx[grid.idx(i, j, k)]*qq_rslt.vx[grid.idx(i, j, k)]
            + qq_rslt.vy[grid.idx(i, j, k)]*qq_rslt.vy[grid.idx(i, j, k)]
            + qq_rslt.vz[grid.idx(i, j, k)]*qq_rslt.vz[grid.idx(i, j, k)] )
        - pii8<Real>*(
            + qq_rslt.bx[grid.idx(i, j, k)]*qq_rslt.bx[grid.idx(i, j, k)]
            + qq_rslt.by[grid.idx(i, j, k)]*qq_rslt.by[grid.idx(i, j, k)]
            + qq_rslt.bz[grid.idx(i, j, k)]*qq_rslt.bz[grid.idx(i, j, k)] )
    )/qq_rslt.ro[grid.idx(i, j, k)];

}

template <typename Real>
struct TimeIntegrator {
    // Disallow copying and assignment since this class manages resources
    TimeIntegrator(const TimeIntegrator&) = delete;
    TimeIntegrator& operator=(const TimeIntegrator&) = delete;

    Model<Real>& model;
    Config& config;
    Time<Real>& time;
    Grid<Real>& grid;
    GridDevice<Real>& grid_d;
    EOS<Real>& eos;
    MHD<Real>& mhd;
    MHDDevice<Real>& mhd_d;    
    MPIManager<Real>& mpi;
    CudaManager<Real>& cuda;
    TimeDevice<Real>& time_d;

    std::unique_ptr<BoundaryConditionBase<Real, MHDCoreDevice<Real>, GridDevice<Real> >> bc;
    ArtificialViscosity<Real> artdiff;

    // Array3D<Real> pr, bb, ht, vb;
    Array3DDevice<Real> pr_d, bb_d, ht_d, vb_d;
    Real cfl_number;
    /// @brief propagation speed fo divergence B
    Real ch_divb;
    /// @brief square of ch_divb;
    Real ch_divb_square;
    /// @brief damping time scape for divergence B
    Real tau_divb;
    
    TimeIntegrator(Model<Real>& model_)
        : model(model_),
          config(model_.config),
          time(model_.time),
          grid(model_.grid_local),
          grid_d(model_.grid_d),
          eos(model_.eos),
          mhd(model_.mhd),
          mhd_d(model_.mhd_d),
          artdiff(model_),
          mpi(model_.mpi),    
          pr_d(grid.i_total, grid.j_total, grid.k_total),
          bb_d(grid.i_total, grid.j_total, grid.k_total),
          ht_d(grid.i_total, grid.j_total, grid.k_total),
          vb_d(grid.i_total, grid.j_total, grid.k_total),
          cuda(model_.cuda),
          time_d(model_.time_d)
          {

            if (config.yaml_obj["boundary_condition"]["boundary_type"].as<std::string>() == "standard") {
                bc = std::make_unique< StandardBoundaryCondition<Real, MHDCoreDevice<Real>, GridDevice<Real> > >(model);
            } else if (config.yaml_obj["boundary_condition"]["boundary_type"].as<std::string>() == "custom") {
                bc = std::make_unique<CustomBoundaryCondition<Real, MHDCoreDevice<Real>, GridDevice<Real> > >(model);
            }
            cfl_number = config.yaml_obj["time_integrator"]["cfl_number"].as<Real>();
          }
        
    
    // core function for MHD time integration
    void update_sc4(const MHDCoreDevice<Real>& qq_orgn, const MHDCoreDevice<Real>& qq_argm, MHDCoreDevice<Real>& qq_rslt, Real dt) {                
        pr_bb_ht_vb_kernel<Real><<<cuda.grid_dim, cuda.block_dim>>>(
            qq_argm, pr_d, bb_d, ht_d, vb_d, grid_d, eos.gm
        );
        CUDA_CHECK(cudaGetLastError());

        update_ro_kernel<Real><<<cuda.grid_dim, cuda.block_dim>>>(
            qq_orgn, qq_argm, qq_rslt,
            grid_d, dt
        );
        CUDA_CHECK(cudaGetLastError());

        update_vx_kernel<Real><<<cuda.grid_dim, cuda.block_dim>>>(
            qq_orgn, qq_argm, qq_rslt,
            pr_d, bb_d,
            grid_d, dt
        );
        CUDA_CHECK(cudaGetLastError());

        update_vy_kernel<Real><<<cuda.grid_dim, cuda.block_dim>>>(
            qq_orgn, qq_argm, qq_rslt,
            pr_d, bb_d,
            grid_d, dt
        );
        CUDA_CHECK(cudaGetLastError());

        update_vz_kernel<Real><<<cuda.grid_dim, cuda.block_dim>>>(
            qq_orgn, qq_argm, qq_rslt,
            pr_d, bb_d,
            grid_d, dt
        );
        CUDA_CHECK(cudaGetLastError());

        update_bx_kernel<Real><<<cuda.grid_dim, cuda.block_dim>>>(
            qq_orgn, qq_argm, qq_rslt,
            grid_d, dt
        );
        CUDA_CHECK(cudaGetLastError());

        update_by_kernel<Real><<<cuda.grid_dim, cuda.block_dim>>>(
            qq_orgn, qq_argm, qq_rslt,
            grid_d, dt
        );
        CUDA_CHECK(cudaGetLastError());

        update_bz_kernel<Real><<<cuda.grid_dim, cuda.block_dim>>>(
            qq_orgn, qq_argm, qq_rslt,
            grid_d, dt
        );
        CUDA_CHECK(cudaGetLastError());

        update_ph_kernel<Real><<<cuda.grid_dim, cuda.block_dim>>>(
            qq_orgn, qq_argm, qq_rslt,
            ch_divb_square, tau_divb,
            grid_d, dt
        );
        CUDA_CHECK(cudaGetLastError());

        update_ei_kernel<Real><<<cuda.grid_dim, cuda.block_dim>>>(
            qq_orgn, qq_argm, qq_rslt,
            pr_d, bb_d, ht_d, vb_d,
            grid_d, dt
        );
    }

    void runge_kutta_4step(){
        MHDCore<Real>& qq     = mhd.qq;
        MHDCore<Real>& qq_argm = mhd.qq_argm;
        MHDCore<Real>& qq_rslt = mhd.qq_rslt;

        // Runge-Kutta 1st step
        update_sc4(mhd_d.qq, mhd_d.qq, mhd_d.qq_rslt, time.dt/4.0);
        mhd_d.qq_argm.copy_from_device(mhd_d.qq_rslt, cuda);
        bc->apply(mhd_d.qq_argm);
        // mhd.mpi_exchange_halo(qq_argm, grid, mpi);

        // Runge-Kutta 2nd step
        update_sc4(mhd_d.qq, mhd_d.qq_argm, mhd_d.qq_rslt, time.dt/3.0);
        mhd_d.qq_argm.copy_from_device(mhd_d.qq_rslt, cuda);
        bc->apply(mhd_d.qq_argm);
        // mhd.mpi_exchange_halo(qq_argm, grid, mpi);

        // Runge-Kutta 3rd step
        update_sc4(mhd_d.qq, mhd_d.qq_argm, mhd_d.qq_rslt, time.dt/2.0);
        mhd_d.qq_argm.copy_from_device(mhd_d.qq_rslt, cuda);
        bc->apply(mhd_d.qq_argm);
        // mhd.mpi_exchange_halo(qq_argm, grid, mpi);

        // Runge-Kutta 4th step
        update_sc4(mhd_d.qq, mhd_d.qq_argm, mhd_d.qq_rslt, time.dt);
        mhd_d.qq.copy_from_device(mhd_d.qq_rslt, cuda);
        bc->apply(mhd_d.qq);
        // mhd.mpi_exchange_halo(qq, grid, mpi);

        mhd_d.qq.copy_to_host(mhd.qq, cuda);
    }
    
    void cfl_condition() {
        cfl_condition_kernel<Real><<<cuda.grid_dim, cuda.block_dim, time_d.shared_mem_size>>>(
            mhd_d.qq, grid_d, time_d.dt_mins_d, cfl_number, eos.gm
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        time_d.copy_to_host();
        time.dt = *std::min_element(time_d.dt_mins_h, time_d.dt_mins_h + time_d.n_blocks);
        Real dt_global;
        MPI_Allreduce(&this->time.dt, &dt_global, 1, mpi_type<Real>(), MPI_MIN, mpi.cart_comm);
        this->time.dt = dt_global;
    }

    void divb_parameters_set() {
        this->ch_divb = 0.8*this->cfl_number*this->grid.min_dxyz/this->time.dt;
        this->ch_divb_square = this->ch_divb*this->ch_divb;
        this->tau_divb = 2.0*this->time.dt;
    }

    void run() {
        // Time integration loop
        this->time.dt = 0.1;

        if (this->config.yaml_obj["base"]["continue"].as<bool>() && fs::exists(this->config.time_save_dir+"n_output.txt")) {
            model.load_state();
        } 

        grid_d.copy_from_host(grid);
        mhd_d.qq.copy_from_host(mhd.qq, cuda);
        model.save_if_needed();


        while (this->time.time < this->time.tend) {

            // basic MHD time integration
            cfl_condition();

            divb_parameters_set();
            runge_kutta_4step();

            // artificial vicosities
            artdiff.characteristic_velocity_eval();

            // x direction
            artdiff.update(grid.dxi, grid_d.dxi, "x");
            mhd_d.qq.copy_from_device(mhd_d.qq_rslt, cuda);
            bc->apply(mhd_d.qq);

            // y direction
            artdiff.update(grid.dyi, grid_d.dyi, "y");
            mhd_d.qq.copy_from_device(mhd_d.qq_rslt, cuda);
            bc->apply(mhd_d.qq);

            // z direction
            artdiff.update(grid.dzi, grid_d.dzi, "z");
            mhd_d.qq.copy_from_device(mhd_d.qq_rslt, cuda);
            bc->apply(mhd_d.qq);

            // Time is update after all procedures
            this->time.update();
            model.save_if_needed();
        }
    }
};

