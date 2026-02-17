#include <random>

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
  Real v_amp;
  Real pr0;
  Real ro_upper;
  Real ro_lower;
  Real g_grav;

  explicit InitialCondition(Config &config, eos::IdealEOS<Real> &eos)
      : eos(eos), v_amp(config["rayleigh_taylor"]["v_amp"].as<Real>()),
        pr0(config["rayleigh_taylor"]["pr0"].as<Real>()),
        ro_upper(config["rayleigh_taylor"]["ro_upper"].as<Real>()),
        ro_lower(config["rayleigh_taylor"]["ro_lower"].as<Real>()),
        g_grav(config["rayleigh_taylor"]["g_grav"].as<Real>()) {}

  // The signature must not be changed as it is called inside miso::mhd::MHD.
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {
    std::mt19937 engine(mpi::rank());
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);

    for (int i = 0; i < grid.i_total; ++i) {
      for (int j = 0; j < grid.j_total; ++j) {
        for (int k = 0; k < grid.k_total; ++k) {
          if (grid.y[j] > 0.0) {
            qq.ro(i, j, k) = ro_upper;
          } else {
            qq.ro(i, j, k) = ro_lower;
          }
          Real pr = pr0 - g_grav * qq.ro(i, j, k) * grid.y[j];
          qq.ei(i, j, k) = pr / (eos.gm - 1.0) / qq.ro(i, j, k);
          qq.vx(i, j, k) = v_amp * dist(engine);
          qq.vy(i, j, k) = v_amp * dist(engine);
          qq.vz(i, j, k) = 0.0;

          qq.bx(i, j, k) = 0.0;
          qq.by(i, j, k) = 0.0;
          qq.bz(i, j, k) = 0.0;
          qq.ph(i, j, k) = 0.0;
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

    if (bc::is_physical_boundary(Direction::Y, Side::INNER, mpi_shape)) {
      bc::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.vy, grid, Sign::Neg, Direction::Y, Side::INNER);
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
      bc::symmetric(btag, qq.vy, grid, Sign::Neg, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.by, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::Y, Side::OUTER);
    }
  }
};

struct SourceTerm {
  Real g_grav;

  SourceTerm(Config &config)
      : g_grav(config["rayleigh_taylor"]["g_grav"].as<Real>()) {}

  /// External force: x-direction
  __host__ __device__ inline Real vx(mhd::FieldsView<const Real>, int, int,
                                     int) const noexcept {
    return 0.0;
  }

  /// External force: y-direction
  __host__ __device__ inline Real vy(mhd::FieldsView<const Real> qq, int i, int j,
                                     int k) const noexcept {
    return -g_grav * qq.ro(i, j, k);
  }

  /// External force: z-direction
  __host__ __device__ inline Real vz(mhd::FieldsView<const Real>, int, int,
                                     int) const noexcept {
    return 0.0;
  }

  /// External heating
  __host__ __device__ inline Real ei(mhd::FieldsView<const Real>, int, int,
                                     int) const noexcept {
    return 0.0;
  }
};

struct Model : public mhd::ModelBase<Model, Real, Backend> {
  eos::IdealEOS<Real> eos;
  InitialCondition ic;
  BoundaryCondition bc;
  SourceTerm src;

  Model(Config &config)
      : ModelBase(config), eos(config), ic(config, eos), bc(mpi_shape),
        src(config) {}
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
