#pragma once
#include "cuda_manager.cuh"
#include <cassert>

template <typename Real> struct MHDCoreDevice {
  Real *ro = nullptr;
  Real *vx = nullptr;
  Real *vy = nullptr;
  Real *vz = nullptr;
  Real *bx = nullptr;
  Real *by = nullptr;
  Real *bz = nullptr;
  Real *ei = nullptr;
  Real *ph = nullptr;

  int i_total, j_total, k_total;

  MHDCoreDevice(const Grid<Real> &grid)
      : i_total(grid.i_total), j_total(grid.j_total), k_total(grid.k_total) {
    CUDA_CHECK(cudaMalloc(&ro, sizeof(Real) * i_total * j_total * k_total));
    CUDA_CHECK(cudaMalloc(&vx, sizeof(Real) * i_total * j_total * k_total));
    CUDA_CHECK(cudaMalloc(&vy, sizeof(Real) * i_total * j_total * k_total));
    CUDA_CHECK(cudaMalloc(&vz, sizeof(Real) * i_total * j_total * k_total));
    CUDA_CHECK(cudaMalloc(&bx, sizeof(Real) * i_total * j_total * k_total));
    CUDA_CHECK(cudaMalloc(&by, sizeof(Real) * i_total * j_total * k_total));
    CUDA_CHECK(cudaMalloc(&bz, sizeof(Real) * i_total * j_total * k_total));
    CUDA_CHECK(cudaMalloc(&ei, sizeof(Real) * i_total * j_total * k_total));
    CUDA_CHECK(cudaMalloc(&ph, sizeof(Real) * i_total * j_total * k_total));
  }

  ~MHDCoreDevice() {}

  void free() {
    if (ro)
      CUDA_CHECK(cudaFree(ro));
    ro = nullptr;
    if (vx)
      CUDA_CHECK(cudaFree(vx));
    vx = nullptr;
    if (vy)
      CUDA_CHECK(cudaFree(vy));
    vy = nullptr;
    if (vz)
      CUDA_CHECK(cudaFree(vz));
    vz = nullptr;
    if (bx)
      CUDA_CHECK(cudaFree(bx));
    bx = nullptr;
    if (by)
      CUDA_CHECK(cudaFree(by));
    by = nullptr;
    if (bz)
      CUDA_CHECK(cudaFree(bz));
    bz = nullptr;
    if (ei)
      CUDA_CHECK(cudaFree(ei));
    ei = nullptr;
    if (ph)
      CUDA_CHECK(cudaFree(ph));
    ph = nullptr;
  }

  MHDCoreDevice(const MHDCoreDevice &) = default;
  MHDCoreDevice &operator=(const MHDCoreDevice &) = default;

  void copy_from_host(const MHDCore<Real> &qq_h, CudaManager<Real> &cuda) {
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaMemcpyAsync(ro, qq_h.ro.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_ro));
    CUDA_CHECK(cudaMemcpyAsync(vx, qq_h.vx.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_vx));
    CUDA_CHECK(cudaMemcpyAsync(vy, qq_h.vy.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_vy));
    CUDA_CHECK(cudaMemcpyAsync(vz, qq_h.vz.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_vz));
    CUDA_CHECK(cudaMemcpyAsync(bx, qq_h.bx.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_bx));
    CUDA_CHECK(cudaMemcpyAsync(by, qq_h.by.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_by));
    CUDA_CHECK(cudaMemcpyAsync(bz, qq_h.bz.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_bz));
    CUDA_CHECK(cudaMemcpyAsync(ei, qq_h.ei.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_ei));
    CUDA_CHECK(cudaMemcpyAsync(ph, qq_h.ph.data(),
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyHostToDevice, cuda.stream_ph));

    cudaDeviceSynchronize();
  }

  void copy_to_host(MHDCore<Real> &qq_h, CudaManager<Real> &cuda) {
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaMemcpyAsync(qq_h.ro.data(), ro,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_ro));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.vx.data(), vx,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_vx));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.vy.data(), vy,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_vy));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.vz.data(), vz,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_vz));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.bx.data(), bx,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_bx));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.by.data(), by,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_by));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.bz.data(), bz,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_bz));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.ei.data(), ei,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_ei));
    CUDA_CHECK(cudaMemcpyAsync(qq_h.ph.data(), ph,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToHost, cuda.stream_ph));

    cudaDeviceSynchronize();
  }

  void copy_from_device(const MHDCoreDevice<Real> &qq_d,
                        CudaManager<Real> &cuda) {
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaMemcpyAsync(ro, qq_d.ro,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_ro));
    CUDA_CHECK(cudaMemcpyAsync(vx, qq_d.vx,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_vx));
    CUDA_CHECK(cudaMemcpyAsync(vy, qq_d.vy,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_vy));
    CUDA_CHECK(cudaMemcpyAsync(vz, qq_d.vz,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_vz));
    CUDA_CHECK(cudaMemcpyAsync(bx, qq_d.bx,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_bx));
    CUDA_CHECK(cudaMemcpyAsync(by, qq_d.by,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_by));
    CUDA_CHECK(cudaMemcpyAsync(bz, qq_d.bz,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_bz));
    CUDA_CHECK(cudaMemcpyAsync(ei, qq_d.ei,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_ei));
    CUDA_CHECK(cudaMemcpyAsync(ph, qq_d.ph,
                               sizeof(Real) * i_total * j_total * k_total,
                               cudaMemcpyDeviceToDevice, cuda.stream_ph));

    cudaDeviceSynchronize();
  }
};

template <typename Real> struct MHDDevice {
  MHDCoreDevice<Real> qq, qq_argm, qq_rslt;
  Real cfl_number;

  MHDDevice(const Grid<Real> &grid, const MHD<Real> &mhd)
      : cfl_number(mhd.cfl_number), qq(grid), qq_argm(grid), qq_rslt(grid) {}

  // destructor (by default GPU memory is freed when MHDDevice is destroyed)
  ~MHDDevice() {}

  void free() {
    qq.free();
    qq_argm.free();
    qq_rslt.free();
  }

  MHDDevice(const MHDDevice &) = default;
  MHDDevice &operator=(const MHDDevice &) = default;
};