#include <memory>

#include <miso/array3d.hpp>
#include <miso/boundary_condition.hpp>
#include <miso/custom_boundary_condition_impl.hpp>
#include <miso/grid.hpp>
#include <miso/mhd.hpp>
#include <miso/model.hpp>

// Strong declaration of the user-defined function.
// Weak declaration is in include/custom_boundary_condition.hpp
template <typename Real>
std::unique_ptr<
    BoundaryConditionBase<Real, MHDCoreDevice<Real>, GridDevice<Real>>>
create_custom_boundary_condition(Model<Real> &model) {
  return std::make_unique<
      CustomBoundaryCondition<Real, MHDCoreDevice<Real>, GridDevice<Real>>>(
      model);
}

// explicit instantiation
template std::unique_ptr<
    BoundaryConditionBase<float, MHDCoreDevice<float>, GridDevice<float>>>
create_custom_boundary_condition(Model<float> &);

template std::unique_ptr<
    BoundaryConditionBase<double, MHDCoreDevice<double>, GridDevice<double>>>
create_custom_boundary_condition(Model<double> &);
