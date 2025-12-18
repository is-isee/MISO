#pragma once

#include <miso/boundary_condition_base.hpp>
#include <miso/mhd_cpu.hpp>
#include <miso/model.hpp>

namespace miso {
namespace bnd {

// This is a weak declaration of the user-defined function.
// An actual implementation should be provided in the problem-specific code. at problems/XXX/custom_boundary_condition_impl.hpp

#ifdef USE_CUDA
template <typename Real>
std::unique_ptr<
    BoundaryConditionBase<Real, mhd::MHDCoreDevice<Real>, GridDevice<Real>>>
create_custom_boundary_condition(Model<Real> &model);
#else
template <typename Real>
std::unique_ptr<BoundaryConditionBase<Real, mhd::MHDCore<Real>, Grid<Real>>>
create_custom_boundary_condition(Model<Real> &model);
#endif

}  // namespace bnd
}  // namespace miso
