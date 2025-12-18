#include <memory>

#include <miso/array3d_cpu.hpp>
#include <miso/boundary_condition_base.hpp>
#include <miso/boundary_condition_core.hpp>
#include <miso/boundary_condition_core_gpu.cuh>
#include <miso/grid_cpu.hpp>
#include <miso/mhd_cpu.hpp>
#include <miso/model.hpp>

#include "custom_boundary_condition_impl.hpp"

namespace miso {

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

}  // namespace miso
