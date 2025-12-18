#pragma once
#include <memory>

#include <miso/boundary_condition_base.hpp>
#include <miso/boundary_condition_core.hpp>
#include <miso/model.hpp>

template <typename Real, typename MHDCoreType, typename GridType>
struct CustomBoundaryCondition
    : public miso::bnd::BoundaryConditionBase<Real, MHDCoreType, GridType> {
  miso::Config &config;
  GridType &grid;
  miso::EOS<Real> &eos;
  miso::MPIManager &mpi;

  CustomBoundaryCondition(miso::Model<Real> &model)
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
    miso::bnd::symmetric<Real>(qq.ro, grid, nullptr, 1.0, miso::bnd::Direction::X,
                               miso::bnd::Side::INNER);
    miso::bnd::symmetric<Real>(qq.ro, grid, nullptr, -1.0,
                               miso::bnd::Direction::X, miso::bnd::Side::OUTER);
  }
};
