#pragma once
#include "config.hpp"

/// @brief Class for equation of state (EOS) parameters
/// @tparam Real
template <typename Real> struct EOS {
  /// @brief ration of specific heats (gamma)
  Real gm;

  /// @brief Constructor for EOS
  /// @param config
  EOS(const Config &config)
      : gm(config.yaml_obj["eos"]["gm"].template as<Real>()) {}
};