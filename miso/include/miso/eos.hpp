#pragma once

#include "config.hpp"
#include "execution.hpp"
#include "mhd.hpp"
#include "utility.hpp"

namespace miso {
namespace eos {

/// @brief Ideal gas equation of state
template <typename Real> struct IdealEOS {
  /// @brief Ratio of specific heats (gamma)
  Real gm;

  /// @brief Construct the EOS from the configuration file
  explicit IdealEOS(const Config &config) : gm(config["eos"]["gm"].as<Real>()) {}

  /// @brief Construct the EOS from the specific heat ratio (gamma)
  explicit IdealEOS(Real gm) : gm(gm) {}

  /// @brief Compute gas pressure from primitive MHD fields.
  /// @note The signature must not be changed as it is called inside mhd::MHD.
  template <typename Backend>
  void gas_pressure(Backend btag, mhd::FieldsView<const Real> qq,
                    Array3DView<Real> pr) const {
    const Real gm_ = gm;  // device lambdas must not capture `this`
    Range1D range{0, qq.size()};
    for_each(
        btag, range,
        MISO_LAMBDA(int i) { pr[i] = (gm_ - Real(1)) * qq.ro[i] * qq.ei[i]; });
  }

  /// @brief Compute speed of sound from primitive MHD fields.
  /// @note The signature must not be changed as it is called inside mhd::MHD.
  template <typename Backend>
  void sound_speed(Backend btag, mhd::FieldsView<const Real> qq,
                   Array3DView<Real> cs) const {
    const Real gm_ = gm;  // device lambdas must not capture `this`
    Range1D range{0, qq.size()};
    for_each(
        btag, range, MISO_LAMBDA(int i) {
          cs[i] = util::sqrt(gm_ * (gm_ - Real(1)) * qq.ei[i]);
        });
  }
};

}  // namespace eos
}  // namespace miso
