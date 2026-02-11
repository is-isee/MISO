#include "common.hpp"

template <typename Real> struct ExternalSources {
  GridView<const Real> grid;
  Real g_grav;

  explicit ExternalSources(Config &config, Grid<Real, Backend> &grid_)
      : grid(grid_.const_view()) {
    g_grav = config["magnetosphere"]["gravitational_acceleration"].as<Real>();
  }

  // External force: x-direction
  // The signature must not be changed as it is called by miso integrator.
  // force is defined in the unit of g/cm^2 s^2 i.e., force per unit volume
  // i.e., acceleration * density
  __host__ __device__ inline Real vx(mhd::FieldsView<const Real> qq, int i, int j,
                                     int k) const noexcept {
    return -qq.ro(i, j, k) * g_grav * grid.x[i] /
           util::pow3(util::sqrt(grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] +
                                 grid.z[k] * grid.z[k]));
  }

  // External force: y-direction
  __host__ __device__ inline Real vy(mhd::FieldsView<const Real> qq, int i, int j,
                                     int k) const noexcept {
    return -qq.ro(i, j, k) * g_grav * grid.y[j] /
           util::pow3(util::sqrt(grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] +
                                 grid.z[k] * grid.z[k]));
  }

  // External force: z-direction
  __host__ __device__ inline Real vz(mhd::FieldsView<const Real> qq, int i, int j,
                                     int k) const noexcept {
    return -qq.ro(i, j, k) * g_grav * grid.z[k] /
           util::pow3(util::sqrt(grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] +
                                 grid.z[k] * grid.z[k]));
  }

  // External heating
  __host__ __device__ inline Real ei(mhd::FieldsView<const Real>, int, int,
                                     int) const noexcept {
    return 0.0;
  }
};
