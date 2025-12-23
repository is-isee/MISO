#pragma once

#include <miso/model.hpp>
#include <miso/utility.hpp>

template <typename Real, typename MHDCoreType, typename GridType> struct Force {
  GridType grid;

  explicit Force(miso::Model<Real> &model)
      :
#ifdef USE_CUDA
        grid(model.grid_d)
#else
        grid(model.grid_local)
#endif
  {
  }

  // force is defined by force per unit volume (i.e., acceleration * density)
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
