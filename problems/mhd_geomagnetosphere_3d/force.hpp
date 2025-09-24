#pragma once

#include "model.hpp"
#include "utility.hpp"

namespace force {
constexpr Real g_grav = 1.35e-6;  // gravitational acceleration (simulation units)
}

#if defined(__CUDACC__)
#define DEVICE __device__
#else
#define DEVICE
#endif

template <typename Real, typename MHDCoreType, typename GridType> struct Force {
  Config &config;
  GridType grid;

  explicit Force(Model<Real> &model)
      : config(model.config),
#ifdef USE_CUDA
        grid(model.grid_d)
#else
        grid(model.grid_local)
#endif
  {
  }
  // force is defined in the unit of g/cm^2 s^2 i.e., force per unit volume
  // i.e., acceleration * density
  DEVICE inline Real x(MHDCoreType &qq, int i, int j, int k) {
    return -qq.ro[grid.idx(i, j, k)] * force::g_grav * grid.x[i] /
           util::pow3(std::sqrt(grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] +
                                grid.z[k] * grid.z[k]));
  }
  DEVICE inline Real y(MHDCoreType &qq, int i, int j, int k) {
    return -qq.ro[grid.idx(i, j, k)] * force::g_grav * grid.y[j] /
           util::pow3(std::sqrt(grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] +
                                grid.z[k] * grid.z[k]));
  }
  DEVICE inline Real z(MHDCoreType &qq, int i, int j, int k) {
    return -qq.ro[grid.idx(i, j, k)] * force::g_grav * grid.z[k] /
           util::pow3(std::sqrt(grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] +
                                grid.z[k] * grid.z[k]));
  }
};
