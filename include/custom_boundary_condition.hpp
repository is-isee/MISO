#pragma once

#include "boundary_condition_base.hpp"
#include "model.hpp"
#ifdef USE_CUDA
#include "boundary_condition_core_gpu.cuh"
#else
#include "boundary_condition_core_cpu.hpp"
#endif

template <typename Real, typename MHDCoreType, typename GridType>
struct CustomBoundaryCondition
    : public BoundaryConditionBase<Real, MHDCoreType, GridType> {
  Config &config;
  Grid<Real> &grid;
  EOS<Real> &eos;
  MHD<Real> &mhd;

  CustomBoundaryCondition(Model<Real> &model)
      : config(model.config), grid(model.grid_local), eos(model.eos),
        mhd(model.mhd) {}

  void apply(MHDCoreType &qq) override {
    // Customized boundary conditions
  }
};