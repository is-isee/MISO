#include "array3d_cpu.hpp"
#include "boundary_condition_base.hpp"
#include "boundary_condition_core.hpp"
#include "boundary_condition_core_gpu.cuh"
#include "custom_boundary_condition_impl.hpp"
#include "grid_cpu.hpp"
#include "mhd_cpu.hpp"
#include "model.hpp"
#include <memory>

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
