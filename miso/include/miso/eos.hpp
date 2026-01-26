#pragma once

#include <miso/config.hpp>

namespace miso {
namespace eos {

/// @brief Ideal gas equation of state
/// @details This class needs to be trivially copyable for device transfer.
template <typename Real> struct IdealEOS {
  /// @brief Ratio of specific heats (gamma)
  Real gm;

  /// @brief Constructor
  /// @param gm_ Ratio of specific heats (gamma)
  explicit IdealEOS(const Config &config) : gm(config["eos"]["gm"].as<Real>()) {}

  /// @brief Compute gas pressure from mass density and specific internal energy
  __host__ __device__ inline Real roeitopr(Real ro, Real ei) const noexcept {
    return (gm - 1.0) * ro * ei;
  }

  /// @brief Compute specific internal energy from mass density and gas pressure
  __host__ __device__ inline Real roprtoei(Real ro, Real pr) const noexcept {
    return pr / (gm - 1.0) / ro;
  }
};

}  // namespace eos
}  // namespace miso
