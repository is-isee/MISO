#include <string>
#include <utility>

#include <miso/boundary_condition.hpp>
#include <miso/config.hpp>
#include <miso/env.hpp>
#include <miso/grid.hpp>
#include <miso/mpi_util.hpp>
#include <miso/rt.hpp>
#include <miso/time.hpp>

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
  Direction direction;
  Side side;
  Real incoming_intensity;
  Real radius;

  static std::pair<Direction, Side> parse_boundary_face(const Config &config) {
    namespace bc = miso::boundary_condition;
    std::string face;

    if (config["searchlight"]["boundary_face"]) {
      face = config["searchlight"]["boundary_face"].as<std::string>();
    } else if (config["searchlight"]["side"]) {
      face = config["searchlight"]["side"].as<std::string>();
    } else {
      face = "x_inner";
    }

    const auto pos = face.find('_');
    if (pos == std::string::npos) {
      throw std::invalid_argument(
          "searchlight.boundary_face must be one of "
          "x_inner/x_outer/y_inner/y_outer/z_inner/z_outer");
    }

    const std::string direction_str = face.substr(0, pos);
    const std::string side_str = face.substr(pos + 1);
    return {bc::string_to_direction(direction_str), bc::string_to_side(side_str)};
  }

  explicit SearchlightBoundaryCondition(const Config &config)
      : direction(parse_boundary_face(config).first),
        side(parse_boundary_face(config).second),
        incoming_intensity(
            config["searchlight"]["incoming_intensity"].as<Real>()),
        radius(config["searchlight"]["radius"].as<Real>()) {}

  void operator()(rt::RT<Real> &solver, const Grid<Real> &grid,
                  const mpi::Shape &mpi_shape) const {
    rt::set_incoming_boundary_on_physical_faces(
        solver, grid, mpi_shape, direction, side, [&](int i, int j, int k) {
          Real rr = 0;
          switch (direction) {
          case Direction::X:
            rr = util::sqrt(util::pow2(grid.y[j]) + util::pow2(grid.z[k]));
            break;
          case Direction::Y:
            rr = util::sqrt(util::pow2(grid.x[i]) + util::pow2(grid.z[k]));
            break;
          case Direction::Z:
            rr = util::sqrt(util::pow2(grid.x[i]) + util::pow2(grid.y[j]));
            break;
          }
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
  Time<Real> time(config);
  time.save();

  const std::string rt_save_dir =
      config.save_dir + config["io"]["rt_save_dir"].as<std::string>();
  util::create_directories(rt_save_dir);
  const auto n_output_digits = config["io"]["n_output_digits"].as<int>();
  const std::string filepath =
      rt_save_dir + "rank_" + util::zfill(mpi::rank(), n_output_digits) + ".bin";
  solver.save(filepath);
}
