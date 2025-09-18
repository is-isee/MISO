#pragma once
#include "boundary_condition_base.hpp"
#include "boundary_condition_core.hpp"
#include "model.hpp"
#include <memory>
#ifdef USE_CUDA
#include "boundary_condition_core_gpu.cuh"
#else
#include "boundary_condition_core_cpu.hpp"
#endif

template <typename Real, typename MHDCoreType, typename GridType>
struct CustomBoundaryCondition
    : public BoundaryConditionBase<Real, MHDCoreType, GridType> {
  Config &config;
  GridType &grid;
  EOS<Real> &eos;
  MPIManager mpi;

  CustomBoundaryCondition(Model<Real> &model)
      : config(model.config),
#ifdef USE_CUDA
        grid(model.grid_d),
#else
        grid(model.grid_local),
#endif
        eos(model.eos), mpi(model.mpi) {
  }

  inline void apply(MHDCoreType &qq) override {
    // write your custom boundary condition here
    // see include/standard_boundary_condition.hpp for reference
  }
};
