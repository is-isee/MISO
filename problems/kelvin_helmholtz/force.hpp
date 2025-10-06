#pragma once

#include "model.hpp"

#if defined(__CUDACC__)
#define DEVICE __device__
#else
#define DEVICE
#endif

template <typename Real, typename MHDCoreType, typename GridType> struct Force {
  GridType grid;

  explicit Force(Model<Real> &model)
      :
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
    // write your custom force here
    return 0.0;
  }
  DEVICE inline Real y(MHDCoreType &qq, int i, int j, int k) {
    // write your custom force here
    return 0.0;
  }
  DEVICE inline Real z(MHDCoreType &qq, int i, int j, int k) {
    // write your custom force here
    return 0.0;
  }
};
