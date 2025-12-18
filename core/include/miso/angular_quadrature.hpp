///
/// @brief Angular quadrature for radiative transfer calculations
///
#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

namespace miso {

namespace rt {

template <class Real> struct AngularQuadrature {
  /// @brief Number of angles
  const int num_rays;

  /// @brief Angle weights
  std::vector<Real> weights;

  /// @brief Angle directions
  std::vector<Real> mu_x, mu_y, mu_z;

  AngularQuadrature(const int num_rays)
      : num_rays(num_rays), weights(num_rays), mu_x(num_rays), mu_y(num_rays),
        mu_z(num_rays) {
    switch (num_rays) {
    case 24:
      // Carlson's A4 quadrature
      compute_quadrature_carlson_a4();
      break;
    case 1:
      // Single ray in positive-z direction
      compute_quadrature_single_positive_z();
      break;
    default:
      throw std::runtime_error("Unsupported number of angles");
      break;
    }
  };

  void compute_quadrature_carlson_a4() {
    const Real mu0 = 1.0 / 3.0;
    const Real mu1 = std::sqrt(1.0 - 2.0 * mu0 * mu0);  // sqrt(7/9)
    const Real ww0 = 1.0 / static_cast<Real>(num_rays);

    weights = {ww0, ww0, ww0, ww0, ww0, ww0, ww0, ww0, ww0, ww0, ww0, ww0,
               ww0, ww0, ww0, ww0, ww0, ww0, ww0, ww0, ww0, ww0, ww0, ww0};
    mu_x = {-mu1, +mu1, -mu1, +mu1, -mu1, +mu1, -mu1, +mu1,
            -mu0, +mu0, -mu0, +mu0, -mu0, +mu0, -mu0, +mu0,
            -mu0, +mu0, -mu0, +mu0, -mu0, +mu0, -mu0, +mu0};
    mu_y = {-mu0, -mu0, +mu0, +mu0, -mu0, -mu0, +mu0, +mu0,
            -mu1, -mu1, +mu1, +mu1, -mu1, -mu1, +mu1, +mu1,
            -mu0, -mu0, +mu0, +mu0, -mu0, -mu0, +mu0, +mu0};
    mu_z = {-mu0, -mu0, -mu0, -mu0, +mu0, +mu0, +mu0, +mu0,
            -mu0, -mu0, -mu0, -mu0, +mu0, +mu0, +mu0, +mu0,
            -mu1, -mu1, -mu1, -mu1, +mu1, +mu1, +mu1, +mu1};
  };

  void compute_quadrature_single_positive_z() {
    weights = {1.0};
    mu_x = {0.0};
    mu_y = {0.0};
    mu_z = {1.0};
  };
};

}  // namespace rt

}  // namespace miso
