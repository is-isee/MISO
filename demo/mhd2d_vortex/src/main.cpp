#include <string>

#include <miso/config.hpp>
#include <miso/env.hpp>
#include <miso/mhd_integrator.hpp>
#include <miso/model.hpp>
#include <miso/types.hpp>
#include <miso/utility.hpp>

using miso::Real, miso::pi;

void initial_condition(miso::Model<Real> &model) {
  auto &qq = model.mhd.qq;
  const auto &grid = model.grid_local;
  const auto &eos = model.eos;

  Real b0 = std::sqrt(4.0 * pi<Real>) / eos.gm;
  Real v0 = 1.0;

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      for (int k = 0; k < grid.k_total; ++k) {
        qq.ro(i, j, k) = 1.0;
        Real pr = 1.0 / eos.gm;
        qq.ei(i, j, k) = pr / (eos.gm - 1.0) / qq.ro(i, j, k);
        qq.vx(i, j, k) = -v0 * std::sin(2.0 * pi<Real> * grid.y[j]);
        qq.vy(i, j, k) = +v0 * std::sin(2.0 * pi<Real> * grid.x[i]);
        qq.vz(i, j, k) = 0.0;
        qq.bx(i, j, k) = -b0 * std::sin(2.0 * pi<Real> * grid.y[j]);
        qq.by(i, j, k) = +b0 * std::sin(4.0 * pi<Real> * grid.x[i]);
        qq.bz(i, j, k) = 0.0;
      }
    }
  }
}

// Periodic boundary condition is applied by MPI communication.
// Be sure to set "periodic" in domain field of config.yaml.
struct EmptyBC {
  explicit EmptyBC(miso::MHD<Real> &mhd) {}

#ifdef __CUDACC__
  void apply(miso::mhd::MHDCoreDevice<Real> &qq) {}
#else
  void apply(miso::mhd::MHDCore<Real> &qq) {}
#endif
};

int main(int argc, char *argv[]) {
  using namespace miso;
  std::string config_dir = CONFIG_DIR;

  Env ctx(argc, argv);
  Config config(config_dir + "config.yaml");
  Model<Real> model(config);
  model.save_metadata();

  initial_condition(model);
  mhd::TimeIntegrator<Real, EmptyBC> time_integrator(model);
  time_integrator.run();
}
