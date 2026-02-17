#include <miso/boundary_condition.hpp>
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
  Real rol, prl, vvl;
  Real ror, prr, vvr;

  explicit InitialCondition(Config &config, eos::IdealEOS<Real> &eos)
      : eos(eos), rol(config["shock_tube"]["rol"].as<Real>()),
        prl(config["shock_tube"]["prl"].as<Real>()),
        vvl(config["shock_tube"]["vvl"].as<Real>()),
        ror(config["shock_tube"]["ror"].as<Real>()),
        prr(config["shock_tube"]["prr"].as<Real>()),
        vvr(config["shock_tube"]["vvr"].as<Real>()) {}

  // The signature must not be changed as it is called inside miso::mhd::MHD.
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {

    for (int k = 0; k < grid.k_total; ++k) {
      for (int j = 0; j < grid.j_total; ++j) {
        for (int i = 0; i < grid.i_total; ++i) {
          qq.vx(i, j, k) = 0.0;
          qq.vy(i, j, k) = 0.0;
          qq.vz(i, j, k) = 0.0;
          qq.bx(i, j, k) = 0.0;
          qq.by(i, j, k) = 0.0;
          qq.bz(i, j, k) = 0.0;
          qq.ph(i, j, k) = 0.0;

          Real xyz;
          if (grid.i_total > 1) {
            xyz = grid.x[i];
          } else if (grid.j_total > 1) {
            xyz = grid.y[j];
          } else if (grid.k_total > 1) {
            xyz = grid.z[k];
          } else {
            throw std::runtime_error(
                "At least one of grid dimensions must be greater than 1.");
          }

          if (xyz < 0.5) {
            qq.ro(i, j, k) = rol;
            qq.ei(i, j, k) = prl / (eos.gm - 1.0) / qq.ro(i, j, k);
            qq.vx(i, j, k) = vvl;
          } else {
            qq.ro(i, j, k) = ror;
            qq.ei(i, j, k) = prr / (eos.gm - 1.0) / qq.ro(i, j, k);
            qq.vx(i, j, k) = vvr;
          }
        }
      }
    }
  }
};

struct BoundaryCondition {
  mpi::Shape &mpi_shape;

  BoundaryCondition(mpi::Shape &mpi_shape) : mpi_shape(mpi_shape) {}

  // The signature must not be changed as it is called by miso integrator.
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {
    namespace bc = miso::boundary_condition;
    Backend btag{};

    if (bc::is_physical_boundary(Direction::X, Side::INNER, mpi_shape)) {
      bc::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.by, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::X, Side::INNER);
    }

    if (bc::is_physical_boundary(Direction::X, Side::OUTER, mpi_shape)) {
      bc::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.by, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::X, Side::OUTER);
    }

    if (bc::is_physical_boundary(Direction::Y, Side::INNER, mpi_shape)) {
      bc::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.by, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::Y, Side::INNER);
    }

    if (bc::is_physical_boundary(Direction::Y, Side::OUTER, mpi_shape)) {
      bc::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.by, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::Y, Side::OUTER);
    }

    if (bc::is_physical_boundary(Direction::Z, Side::INNER, mpi_shape)) {
      bc::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.by, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::Z, Side::INNER);
    }

    if (bc::is_physical_boundary(Direction::Z, Side::OUTER, mpi_shape)) {
      bc::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.by, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::Z, Side::OUTER);
    }
  }
};

struct Model : public mhd::ModelBase<Model, Real, Backend> {
  eos::IdealEOS<Real> eos;
  InitialCondition ic;
  BoundaryCondition bc;
  mhd::EmptySourceTerm<Real> src;

  Model(Config &config)
      : ModelBase(config), eos(config), ic(config, eos), bc(mpi_shape), src() {}
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
