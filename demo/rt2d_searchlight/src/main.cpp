#include <string>

#include <miso/config.hpp>
#include <miso/env.hpp>
#include <miso/grid.hpp>
#include <miso/mpi_util.hpp>
#include <miso/rt.hpp>

using namespace miso;

using Real = float;

template <typename Real>
void setup(rt::RT<Real> &solver, const Grid<Real> &grid, const Config &config) {
  const Real abs_coeff = config["searchlight"]["abs_coeff"].as<Real>();
  const Real src_func = config["searchlight"]["src_func"].as<Real>();

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      for (int k = 0; k < grid.k_total; ++k) {
        solver.abs_coeff(i, j, k) = abs_coeff;
        solver.src_func(i, j, k) = src_func;
      }
    }
  }

  util::clear_array(solver.rint);
  util::clear_array(solver.rint_old);
}

struct SearchlightBoundaryCondition {
  Real incoming_intensity;
  Real radius;

  explicit SearchlightBoundaryCondition(const Config &config)
      : incoming_intensity(
            config["searchlight"]["incoming_intensity"].as<Real>()),
        radius(config["searchlight"]["radius"].as<Real>()) {}

  void operator()(rt::RT<Real> &solver, const Grid<Real> &grid,
                  const mpi::Shape &mpi_shape) const {
    rt::set_incoming_boundary_on_physical_faces(
        solver, grid, mpi_shape, Direction::X, Side::INNER,
        [&](int, int j, int k) {
          const Real rr =
              util::sqrt(util::pow2(grid.y[j]) + util::pow2(grid.z[k]));
          return (rr <= radius) ? incoming_intensity : Real(0);
        });
  }
};

int main(int argc, char **argv) {
  Env env(argc, argv);

  auto config_path = parse_config_filepath(argc, argv);
  Config config(config_path.value_or("./config.yaml"));

  mpi::Shape mpi_shape(config);
  Grid<Real> grid(config, mpi_shape);
  rt::RT<Real> solver(grid, config["rt"]["num_rays"].as<int>());

  setup(solver, grid, config);
  SearchlightBoundaryCondition bc(config);
  solver.solve(grid, mpi_shape, config["rt"]["tolerance"].as<Real>(),
               config["rt"]["max_iters"].as<int>(), bc);

  config.save();
  grid.save(config);
  mpi_shape.save();

  const std::string rt_save_dir =
      config.save_dir + config["rt"]["save_dir"].as<std::string>();
  util::create_directories(rt_save_dir);
  const std::string filepath =
      rt_save_dir + "rank_" + util::zfill(mpi::rank(), 6) + ".bin";
  solver.save(filepath);
}
