#include "array3d_cpu.hpp"
#include "boundary_condition_base.hpp"
#include "boundary_condition_core.hpp"
#include "boundary_condition_core_cpu.hpp"
#include "custom_boundary_condition_impl.hpp"
#include "grid_cpu.hpp"
#include "mhd_cpu.hpp"
#include "model.hpp"
#include <memory>

//////////////////////////////////////////////
// Users do not need to modify the code below
//////////////////////////////////////////////

// Strong declaration of the user-defined function.
// Weak declaration is in include/custom_boundary_condition.hpp
template <typename Real>
std::unique_ptr<BoundaryConditionBase<Real, MHDCore<Real>, Grid<Real>>>
create_custom_boundary_condition(Model<Real> &model) {
  return std::make_unique<
      CustomBoundaryCondition<Real, MHDCore<Real>, Grid<Real>>>(model);
}

// explicit instantiation
template std::unique_ptr<
    BoundaryConditionBase<float, MHDCore<float>, Grid<float>>>
create_custom_boundary_condition(Model<float> &);

template std::unique_ptr<
    BoundaryConditionBase<double, MHDCore<double>, Grid<double>>>
create_custom_boundary_condition(Model<double> &);
