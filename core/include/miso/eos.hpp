#pragma once
#include <miso/config.hpp>

namespace miso {
namespace eos {

/// @brief Ideal gas equation of state
template <typename Real> struct IdealEOS {
  /// @brief Ratio of specific heats (gamma)
  Real gm;

  IdealEOS(const Config &config) : gm(config["eos"]["gm"].template as<Real>()) {}
};

}  // namespace eos
}  // namespace miso
