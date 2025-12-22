#pragma once

#include <unordered_map>

#include <miso/boundary_condition.hpp>
#include <miso/model.hpp>

namespace miso {
namespace bnd {

template <typename Real, typename MHDCoreType, typename GridType>
struct StandardBoundaryCondition {
  Config &config;
  GridType &grid;
  EOS<Real> &eos;
  MPITopology &mpi;

  StandardBoundaryCondition(Model<Real> &model)
      : config(model.config),
#ifdef USE_CUDA
        grid(model.grid_d),
#else
        grid(model.grid_local),
#endif
        eos(model.eos), mpi(model.mpi) {
  }

  void apply(MHDCoreType &qq) {
    // Apply boundary conditions to the MHD core variables
#ifdef USE_CUDA
    std::unordered_map<std::string, std::reference_wrapper<Real *>> variables = {
        {"ro", qq.ro}, {"vx", qq.vx}, {"vy", qq.vy}, {"vz", qq.vz}, {"bx", qq.bx},
        {"by", qq.by}, {"bz", qq.bz}, {"ei", qq.ei}, {"ph", qq.ph}};
#else
    std::unordered_map<std::string, std::reference_wrapper<Array3D<Real>>>
        variables = {{"ro", qq.ro}, {"vx", qq.vx}, {"vy", qq.vy},
                     {"vz", qq.vz}, {"bx", qq.bx}, {"by", qq.by},
                     {"bz", qq.bz}, {"ei", qq.ei}, {"ph", qq.ph}};
#endif

    const auto &bc_yaml = config.yaml_obj["boundary_condition"];
    const auto &periodic_flags = bc_yaml["periodic"];

    constexpr std::array<Direction, 3> directions = {Direction::X, Direction::Y,
                                                     Direction::Z};
    constexpr std::array<Side, 2> sides = {Side::INNER, Side::OUTER};

    // Loop over all MHD variable arrays stored in the `variables` map.
    // Each map entry consists of:
    //   - `name`  : the variable name as a string (e.g., "ro", "vx")
    //   - `array` : a reference to the corresponding Array3D<Real> object
    for (const auto &[name, array] : variables) {
      for (const auto &direction : directions) {
        std::vector<std::string> conditions;
        conditions = bc_yaml[name][direction_to_string(direction)]
                         .template as<std::vector<std::string>>();
        for (const auto &side : sides) {
          const std::string method = conditions[static_cast<int>(side)];
          if (is_physical_boundary<Real>(direction, side, mpi)) {
            if (method == "symmetric") {
              symmetric<Real>(array.get(), grid, nullptr, 1.0, direction, side);
            } else if (method == "antisymmetric") {
              symmetric<Real>(array.get(), grid, nullptr, -1.0, direction, side);
            } else if (method == "periodic") {
              // Periodic boundary condition is applied by MPI communication.
              // Be sure to set "periodic" in domain field in config.yaml.
            } else {
              throw std::runtime_error("Unknown boundary condition method: " +
                                       method);
            }
          }
        }
      }
    }
  }
};

}  // namespace bnd
}  // namespace miso
