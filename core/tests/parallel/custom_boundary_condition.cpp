#include <memory>

#include <miso/array3d_cpu.hpp>
#include <miso/boundary_condition_base.hpp>
#include <miso/boundary_condition_core.hpp>
#include <miso/grid.hpp>
#include <miso/mhd.hpp>
#include <miso/model.hpp>

#include "custom_boundary_condition_impl.hpp"

// Strong declaration of the user-defined function.
// Weak declaration is in include/custom_boundary_condition.hpp
template <typename Real>
std::unique_ptr<miso::bnd::BoundaryConditionBase<Real, miso::mhd::MHDCore<Real>,
                                                 miso::Grid<Real>>>
create_custom_boundary_condition(miso::Model<Real> &model) {
  return std::make_unique<
      CustomBoundaryCondition<Real, miso::mhd::MHDCore<Real>, miso::Grid<Real>>>(
      model);
}

// explicit instantiation
template std::unique_ptr<miso::bnd::BoundaryConditionBase<
    float, miso::mhd::MHDCore<float>, miso::Grid<float>>>
create_custom_boundary_condition(miso::Model<float> &);

template std::unique_ptr<miso::bnd::BoundaryConditionBase<
    double, miso::mhd::MHDCore<double>, miso::Grid<double>>>
create_custom_boundary_condition(miso::Model<double> &);
