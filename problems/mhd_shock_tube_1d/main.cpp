#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "config.hpp"
#include "constants.hpp"
#include "model.hpp"
#include "types.hpp"
#include "utility.hpp"

#ifdef USE_CUDA
#include "time_integrator_gpu.cuh"
#else
#include "time_integrator_cpu.hpp"
#endif

using util::pow2;

template <typename Real> struct Init {
  Real xm;
  Real bm;
  Real rol, prl, vvl, bpl;
  Real ror, prr, vvr, bpr;
};

template <typename Real>
void initial_condition(Model<Real> &model, const Init<Real> &init,
                       std::string direction);

int main() {
  std::string config_dir = CONFIG_DIR;
  // initial condition setting
  Init<Real> init;
  init.bm = 0.75 * std::sqrt(4 * pi<Real>);
  init.rol = 1.0;
  init.prl = 1.0;
  init.bpl = +1.0 * std::sqrt(4 * pi<Real>);
  init.vvl = 0.0;
  init.ror = 0.125;
  init.prr = 0.1;
  init.bpr = -1.0 * std::sqrt(4 * pi<Real>);
  init.vvr = 0.0;

  std::vector<std::string> directions = {"x", "y", "z"};
  MPIManager<Real> mpi;

  int m = 0;
  for (const auto &direction : directions) {
    {
      if (mpi.myrank == 0) {
        std::cout << "####################################################"
                  << std::endl;
        std::cout << "Running simulation in direction: " << direction
                  << std::endl;
        std::cout << "####################################################"
                  << std::endl;
      }
      Config config(config_dir + "config_" + direction + ".yaml", mpi);
      mpi.setup_mpi(config.yaml_obj);
      Model<Real> model(config);
      model.save_metadata();

      initial_condition<Real>(model, init, direction);
      TimeIntegrator<Real> time_integrator(model);
      time_integrator.run();
    }
    m++;
  }

  return 0;
}

template <typename Real>
void initial_condition(Model<Real> &model, const Init<Real> &init,
                       std::string direction) {
  MHDCore<Real> &qq = model.mhd.qq;
  const auto &grid = model.grid_local;
  const auto &eos = model.eos;

  std::reference_wrapper<Array3D<Real>> b_main = qq.bx;
  std::reference_wrapper<Array3D<Real>> b_perp = qq.by;

  if (direction == "x") {
    b_main = qq.bx;
    b_perp = qq.by;
  } else if (direction == "y") {
    b_main = qq.by;
    b_perp = qq.bz;
  } else if (direction == "z") {
    b_main = qq.bz;
    b_perp = qq.bx;
  } else {
    std::cerr << "Invalid direction: " << direction << std::endl;
    return;
  }

  Real xyz;

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      for (int k = 0; k < grid.k_total; ++k) {

        qq.vy(i, j, k) = 0.0;
        qq.vz(i, j, k) = 0.0;
        qq.ph(i, j, k) = 0.0;

        qq.vx(i, j, k) = 0.0;

        b_main(i, j, k) = 0.75 * std::sqrt(4 * pi<Real>);

        if (direction == "x") {
          xyz = grid.x[i];
        } else if (direction == "y") {
          xyz = grid.y[j];
        } else if (direction == "z") {
          xyz = grid.z[k];
        } else {
          std::cerr << "Invalid direction: " << direction << std::endl;
          return;
        }

        if (xyz < 0.5) {
          qq.ro(i, j, k) = init.rol;
          qq.ei(i, j, k) = init.prl / (eos.gm - 1.0) / qq.ro(i, j, k);
          qq.vx(i, j, k) = init.vvl;
          b_perp(i, j, k) = init.bpl;
        } else {
          qq.ro(i, j, k) = init.ror;
          qq.ei(i, j, k) = init.prr / (eos.gm - 1.0) / qq.ro(i, j, k);
          qq.vx(i, j, k) = init.vvr;
          b_perp(i, j, k) = init.bpr;
        }
      }
    }
  }
}