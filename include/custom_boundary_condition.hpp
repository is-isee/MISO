#pragma once
#include "boundary_condition_base.hpp"
#include "mhd_cpu.hpp"
#include "model.hpp"

// This is a weak declaration of the user-defined function.
// An actual implementation should be provided in the problem-specific code. at problems/XXX/custom_boundary_condition.cpp

#ifdef USE_CUDA
template <typename Real>
std::unique_ptr<
    BoundaryConditionBase<Real, MHDCoreDevice<Real>, GridDevice<Real>>>
create_custom_boundary_condition(Model<Real> &model);
#else
template <typename Real>
std::unique_ptr<BoundaryConditionBase<Real, MHDCore<Real>, Grid<Real>>>
create_custom_boundary_condition(Model<Real> &model);
#endif
