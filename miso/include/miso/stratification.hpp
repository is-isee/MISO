///
/// @brief One-dimensional stratification read from a table (r2d2plus-table)
///
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "eos_table.hpp"

namespace miso {

/// @brief Stratification as a function of the height z, read from a table of
/// kind `stratification_z` (for example,
/// `input_data/stratification_miso.tbl`, see `input_data/README.md`).
/// @details The table has the 1D arrays z (increasing, not uniform), ro, en,
/// pr, te, se and gx in CGS units. `at(z)` interpolates linearly in z: ln ro,
/// ln ei, ln pr and ln te for the positive quantities, and se and gx as they
/// are. It throws if z is outside the table. Used on the host, for example in
/// an initial condition.
template <typename Real> struct Stratification {
  /// @brief Values at a height
  struct State {
    /// @brief Density [g/cm^3]
    Real ro;
    /// @brief Internal energy per unit mass [erg/g]
    Real ei;
    /// @brief Gas pressure [dyn/cm^2]
    Real pr;
    /// @brief Temperature [K]
    Real te;
    /// @brief Specific entropy [erg/g/K]
    Real se;
    /// @brief Magnitude of the gravitational acceleration [cm/s^2], toward -z
    Real gx;
  };

  /// @brief Height [cm] of the table points (increasing)
  std::vector<Real> z;
  /// @brief Values at the table points
  std::vector<Real> ro, ei, pr, te, se, gx;

  explicit Stratification(const fs::path &path)
      : Stratification(eos::TableFile(path)) {}

  explicit Stratification(const eos::TableFile &file) {
    if (file.kind != "stratification_z") {
      throw std::runtime_error(
          "Expected a table of kind stratification_z, got: " + file.kind);
    }
    const auto it = file.shapes.find("z");
    if (it == file.shapes.end() || it->second.size() != 1 || it->second[0] < 2) {
      throw std::runtime_error("Table has no 1D array 'z' of length >= 2");
    }
    const int n = it->second[0];
    const auto copy = [&](const std::string &name, std::vector<Real> &out) {
      const auto &v = file.get(name, {n});
      out.assign(v.begin(), v.end());
    };
    copy("z", z);
    copy("ro", ro);
    copy("en", ei);
    copy("pr", pr);
    copy("te", te);
    copy("se", se);
    copy("gx", gx);
    for (int i = 0; i + 1 < n; ++i) {
      if (!(z[i] < z[i + 1])) {
        throw std::runtime_error("Table axis 'z' is not increasing");
      }
    }
  }

  /// @brief Values at the height `zz` [cm], interpolated linearly in z.
  State at(Real zz) const {
    if (!(z.front() <= zz && zz <= z.back())) {
      throw std::out_of_range("Height " + std::to_string(zz) +
                              " is outside the stratification table");
    }
    // i: the last point with z[i] <= zz, but at most n - 2
    const auto upper = std::upper_bound(z.begin(), z.end(), zz);
    const int i = std::min(static_cast<int>(upper - z.begin()) - 1,
                           static_cast<int>(z.size()) - 2);
    const Real w = (zz - z[i]) / (z[i + 1] - z[i]);
    const auto lin = [&](const std::vector<Real> &v) {
      return (Real(1) - w) * v[i] + w * v[i + 1];
    };
    const auto log_lin = [&](const std::vector<Real> &v) {
      return std::exp((Real(1) - w) * std::log(v[i]) + w * std::log(v[i + 1]));
    };
    return {log_lin(ro), log_lin(ei), log_lin(pr), log_lin(te), lin(se), lin(gx)};
  }
};

}  // namespace miso
