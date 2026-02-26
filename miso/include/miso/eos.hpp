#pragma once

#include "config.hpp"
#include "utility.hpp"

namespace miso {
namespace eos {

/// @brief Ideal gas equation of state
/// @details This class needs to be trivially copyable for device transfer.
template <typename Real> struct IdealEOS {
  /// @brief Ratio of specific heats (gamma)
  Real gm;

  /// @brief Construct the EOS from the configuration file
  explicit IdealEOS(const Config &config) : gm(config["eos"]["gm"].as<Real>()) {}

  /// @brief Construct the EOS from the specific heat ratio (gamma)
  explicit IdealEOS(Real gm) : gm(gm) {}

  /// @brief Compute gas pressure from mass density and specific internal energy
  __host__ __device__ inline Real roeitopr(Real ro, Real ei) const noexcept {
    return (gm - 1.0) * ro * ei;
  }

  /// @brief Compute specific internal energy from mass density and gas pressure
  __host__ __device__ inline Real roprtoei(Real ro, Real pr) const noexcept {
    return pr / (gm - 1.0) / ro;
  }

  /// @brief Compute adiabatic speed of sound from mass density and specific internal energy
  __host__ __device__ inline Real roeitocs(Real /*ro*/, Real ei) const noexcept {
    return util::sqrt(gm * (gm - 1.0) * ei);
  }
};

}  // namespace eos
}  // namespace miso
