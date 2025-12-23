#pragma once

#include <cassert>
#include <vector>

#include <miso/array3d_cpu.hpp>
#include <miso/array4d_cpu.hpp>
#include <miso/grid_cpu.hpp>
#include <miso/mpi_manager.hpp>
#include <miso/mpi_types.hpp>
#include <miso/time.hpp>
#include <miso/utility.hpp>

namespace miso {
namespace mhd {

template <typename Real> struct MHDCore {
  Array3D<Real> ro, vx, vy, vz, bx, by, bz, ei, ph;

  MHDCore(int i_size, int j_size, int k_size)
      : ro(i_size, j_size, k_size), vx(i_size, j_size, k_size),
        vy(i_size, j_size, k_size), vz(i_size, j_size, k_size),
        bx(i_size, j_size, k_size), by(i_size, j_size, k_size),
        bz(i_size, j_size, k_size), ei(i_size, j_size, k_size),
        ph(i_size, j_size, k_size) {}

  void copy_from(const MHDCore &other) {
    ro.copy_from(other.ro);
    vx.copy_from(other.vx);
    vy.copy_from(other.vy);
    vz.copy_from(other.vz);
    bx.copy_from(other.bx);
    by.copy_from(other.by);
    bz.copy_from(other.bz);
    ei.copy_from(other.ei);
    ph.copy_from(other.ph);
  }
};

}  // namespace mhd
}  // namespace miso
