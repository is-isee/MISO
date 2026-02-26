#include <miso/mhd_model_base.hpp>

using namespace miso;

using Real = float;

#ifdef USE_CUDA
using Backend = backend::CUDA;
#else
using Backend = backend::Host;
#endif

struct InitialCondition {
  eos::IdealEOS<Real> &eos;

  explicit InitialCondition(eos::IdealEOS<Real> &eos) : eos(eos) {}

  // The signature must not be changed as it is called inside miso::mhd::MHD.
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {
    const Real pr = 1.0 / eos.gm;
    const Real b0 = util::sqrt(4.0 * pi<Real>) / eos.gm;
    const Real v0 = 1.0;

    for (int k = 0; k < grid.k_total; ++k) {
      for (int j = 0; j < grid.j_total; ++j) {
        for (int i = 0; i < grid.i_total; ++i) {
          qq.ro(i, j, k) = 1.0;
          qq.ei(i, j, k) = pr / (eos.gm - 1.0) / qq.ro(i, j, k);
          qq.vx(i, j, k) = -v0 * util::sin(2.0 * pi<Real> * grid.y[j]);
          qq.vy(i, j, k) = +v0 * util::sin(2.0 * pi<Real> * grid.x[i]);
          qq.vz(i, j, k) = 0.0;
          qq.bx(i, j, k) = -b0 * util::sin(2.0 * pi<Real> * grid.y[j]);
          qq.by(i, j, k) = +b0 * util::sin(4.0 * pi<Real> * grid.x[i]);
          qq.bz(i, j, k) = 0.0;
          qq.ph(i, j, k) = 0.0;
        }
      }
    }
  }
};

struct Model : public mhd::ModelBase<Model, Real, Backend> {
  eos::IdealEOS<Real> eos;
  InitialCondition ic;
  mhd::EmptyBoundaryCondition<Real> bc;
  mhd::EmptySourceTerm<Real> src;

  Model(Config &config) : ModelBase(config), eos(config), ic(eos), bc(), src() {}
};

int main(int argc, char **argv) {
  // Initialize MPI and CUDA environments
  Env env(argc, argv);

  // Read configuration file
  auto config_path = parse_config_filepath(argc, argv);
  Config config(config_path.value_or("./config.yaml"));

  // Run simulation
  Model model(config);
  model.run();
}
