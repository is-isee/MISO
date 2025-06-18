#pragma once

#include <cmath>

#include <cmath>  // M_PI 用

template <typename Real>
inline constexpr Real inv12 = static_cast<Real>(1.0 / 12.0);

template <typename Real>
inline constexpr Real pi = static_cast<Real>(M_PI);

template <typename Real>
inline constexpr Real pii4 = static_cast<Real>(1.0 / (4.0 * M_PI)); // 1 / 4π

template <typename Real>
inline constexpr Real pii8 = static_cast<Real>(1.0 / (8.0 * M_PI)); // 1 / 8π