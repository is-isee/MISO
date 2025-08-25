#pragma once

/// @brief Base class for boundary conditions in MHD simulations
/// @tparam Real Type of the data (Real)
template <typename Real, typename MHDCoreType, typename GridType>
class BoundaryConditionBase {
public:
  virtual void apply(MHDCoreType &qq) = 0;
  virtual ~BoundaryConditionBase() = default;
};