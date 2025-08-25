#pragma once

#include "boundary_condition_base.hpp"
#include "model.hpp"
#include <unordered_map>
#ifdef USE_CUDA
#include "boundary_condition_core_gpu.cuh"
#else
#include "boundary_condition_core_cpu.hpp"
#endif

template <typename Real, typename MHDCoreType, typename GridType>
struct StandardBoundaryCondition
    : public BoundaryConditionBase<Real, MHDCoreType, GridType> {
  Config &config;
  GridType &grid;
  EOS<Real> &eos;
  MPIManager<Real> &mpi;

  StandardBoundaryCondition(Model<Real> &model)
      : config(model.config),
#ifdef USE_CUDA
        grid(model.grid_d),
#else
        grid(model.grid_global),
#endif
        eos(model.eos), mpi(model.mpi) {
  }

  void apply(MHDCoreType &qq) override {
    // Apply boundary conditions to the MHD core variables
#ifdef USE_CUDA
    std::unordered_map<std::string, std::reference_wrapper<Real *>> variables = {
        {"ro", qq.ro}, {"vx", qq.vx}, {"vy", qq.vy},
        {"vz", qq.vz}, {"bx", qq.bx}, {"by", qq.by},
        {"bz", qq.bz}, {"ei", qq.ei}, { "ph", qq.ph }};
#else
    std::unordered_map<std::string, std::reference_wrapper<Array3D<Real>>>
        variables = {{"ro", qq.ro}, {"vx", qq.vx}, {"vy", qq.vy},
                     {"vz", qq.vz}, {"bx", qq.bx}, {"by", qq.by},
                     {"bz", qq.bz}, {"ei", qq.ei}, {"ph", qq.ph}};
#endif

    const auto &bc_yaml = config.yaml_obj["boundary_condition"];
    const auto &periodic_flags = bc_yaml["periodic"];

    constexpr std::array<bnd::Direction, 3> directions = {
        bnd::Direction::X, bnd::Direction::Y, bnd::Direction::Z};
    constexpr std::array<bnd::Side, 2> sides = {bnd::Side::INNER,
                                                bnd::Side::OUTER};

    // Loop over all MHD variable arrays stored in the `variables` map.
    // Each map entry consists of:
    //   - `name`  : the variable name as a string (e.g., "ro", "vx")
    //   - `array` : a reference to the corresponding Array3D<Real> object
    for (const auto &[name, array] : variables) {
      for (const auto &direction : directions) {
        bool is_periodic = periodic_flags[bnd::direction_to_string(direction)]
                               .template as<bool>();
        std::vector<std::string> conditions;
        if (is_periodic) {
          conditions = {"periodic", "periodic"};
        } else {
          conditions = bc_yaml[name][bnd::direction_to_string(direction)]
                           .template as<std::vector<std::string>>();
        }
        for (const auto &side : sides) {
          const std::string method = conditions[static_cast<int>(side)];
          if (bnd::is_physical_boundary<Real>(direction, side, mpi)) {
            if (method == "symmetric") {
              bnd::symmetric<Real>(array.get(), grid, nullptr, 1.0, direction,
                                   side);
            } else if (method == "ro_symmetric") {
#ifdef USE_CUDA
              bnd::symmetric<Real>(array.get(), grid, qq.ro, 1.0, direction,
                                   side);
#else
              bnd::symmetric<Real>(array.get(), grid, &qq.ro, 1.0, direction,
                                   side);
#endif
            } else if (method == "antisymmetric") {
              bnd::symmetric<Real>(array.get(), grid, nullptr, -1.0, direction,
                                   side);
            } else if (method == "ro_antisymmetric") {
#ifdef USE_CUDA
              bnd::symmetric<Real>(array.get(), grid, qq.ro, -1.0, direction,
                                   side);
#else
              bnd::symmetric<Real>(array.get(), grid, &qq.ro, -1.0, direction,
                                   side);
#endif
            } else if (method == "periodic") {
              bnd::periodic<Real>(array.get(), grid, direction, side);
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