////
//// Searchlight problem of radiative transfer equation
////

// #include <cmath>
// #include <cstdlib>
// #include <filesystem>
// #include <iostream>
// #include <random>
// #include <string>
// #include <vector>

#include "boundary_condition_core.hpp"
#include "config.hpp"
#include "model.hpp"
#include "mpi_manager.hpp"
#include "radiative_transfer_cpu.hpp"
#include "types.hpp"
#include "utility.hpp"

namespace prm {
static constexpr Real abs_coeff = 1.0;
static constexpr Real src_func = 0.5;
static constexpr Real rint_incoming = 1.0;
static constexpr Real rint_radius = 0.1;
}  // namespace prm

template <typename Real> void setup(Model<Real> &model) {
  auto &rt = model.rt;
  const auto &grid = model.grid_local;
  const auto &eos = model.eos;

  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      for (int k = 0; k < grid.k_total; ++k) {
        rt.abs_coeff(i, j, k) = prm::abs_coeff;
        rt.src_func(i, j, k) = prm::src_func;
      }
    }
  }
}

template <typename Real>
void impose_incoming_ray(RT<Real> &rt, const Grid<Real> &grid,
                         const bnd::Direction direction, const bnd::Side side) {
  using bnd::Direction;
  using bnd::Side;
  const auto &ang_quad = rt.ang_quad;
  const auto ib0 = rt.ib0;
  const auto ib1 = rt.ib1;
  const auto jb0 = rt.jb0;
  const auto jb1 = rt.jb1;
  const auto kb0 = rt.kb0;
  const auto kb1 = rt.kb1;

  int i, j, k;
  for (int i_ray = 0; i_ray < ang_quad.num_rays; ++i_ray) {
    if (direction == Direction::X) {
      if (side == Side::INNER && ang_quad.mu_x[i_ray] > 0.0) {
        i = ib0;
      } else if (side == Side::OUTER && ang_quad.mu_x[i_ray] < 0.0) {
        i = ib1;
      } else {
        continue;
      }
      for (j = jb0; j <= jb1; ++j) {
        for (k = kb0; k <= kb1; ++k) {
          const Real r = std::sqrt(grid.y[j] * grid.y[j] + grid.z[k] * grid.z[k]);
          rt.rint(i_ray, i, j, k) =
              (r <= prm::rint_radius) ? prm::rint_incoming : 0.0;
        }
      }
    }
    if (ang_quad.mu_x[i_ray] < 0.0) {
      // To be implemented
    }
  }
}

struct SearchlightBoundaryCondition {
  const Config config;

  explicit SearchlightBoundaryCondition(const Config &config_)
      : config(config_) {}

  template <typename Real>
  void operator()(RT<Real> &rt, const Grid<Real> &grid,
                  const MPIManager<Real> &mpi) const {
    const auto &periodic_flags =
        config.yaml_obj["boundary_condition"]["periodic"];

    constexpr std::array<bnd::Direction, 3> directions = {
        bnd::Direction::X, bnd::Direction::Y, bnd::Direction::Z};
    constexpr std::array<bnd::Side, 2> sides = {bnd::Side::INNER,
                                                bnd::Side::OUTER};
    for (const auto &direction : directions) {
      bool is_periodic =
          periodic_flags[bnd::direction_to_string(direction)].template as<bool>();
      for (const auto &side : sides) {
        if (bnd::is_physical_boundary<Real>(direction, side, mpi)) {
          impose_incoming_ray<Real>(rt, grid, direction, side);
        }
      }
    }
  }
};

int main() {
  std::string config_dir = CONFIG_DIR;

  MPIManager<Real> mpi_manager;
  Config config(config_dir + "config.yaml", mpi_manager);
  mpi_manager.setup_mpi(config.yaml_obj);
  Model<Real> model(config);
  model.save_metadata();

  setup<Real>(model);
  SearchlightBoundaryCondition bc(config);
  const Real tolerance = 1.e-5;
  const int max_iters = 100;
  model.rt.solve(model.grid_local, model.mpi, tolerance, max_iters, bc);

  return 0;
}
