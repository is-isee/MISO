///
/// @brief Radiative heating as a source term of the MHD solver
/// @details The RT is solved on the CPU. With the CUDA backend, rho and e are
/// copied to the host, and Q_rad back to the device, every time `update` is
/// called.
///
#pragma once

#include <stdexcept>

#include "array3d.hpp"
#include "boundary_condition.hpp"
#include "config.hpp"
#include "eos_table.hpp"
#include "grid.hpp"
#include "mhd_fields.hpp"
#include "mpi_util.hpp"
#include "rt_heating.hpp"
#include "rt_host.hpp"
#include "utility.hpp"

namespace miso {
namespace rt {

/// @brief Source term of the MHD solver that adds Q_rad to the energy equation.
/// @details Holds a view of `Radiation::qrad`. It is valid while that
/// Radiation (or the object it was moved into) lives and is not assigned anew.
template <typename Real> struct RadiativeHeatingSource {
  Array3DView<const Real> qrad;
  __host__ __device__ Real ei(mhd::FieldsView<const Real>, int i, int j,
                              int k) const {
    return qrad(i, j, k);
  }
};

/// @brief Incoming intensity at the z boundaries: 0 at the top, and S of the
/// lowest RT point at the bottom (thermalized deep layer).
struct VerticalBoundary {
  template <typename Real>
  void operator()(RT<Real> &rt, const Grid<Real> &grid,
                  const mpi::Shape &mpi_shape) const {
    set_incoming_boundary_on_physical_faces(
        rt, grid, mpi_shape, Direction::Z, Side::OUTER,
        [](int, int, int) { return Real(0); });
    set_incoming_boundary_on_physical_faces(
        rt, grid, mpi_shape, Direction::Z, Side::INNER,
        [&rt](int i, int j, int k) { return rt.src_func(i, j, k); });
  }
};

///
/// @brief RT and Q_rad for the MHD solver, with the opacity and temperature
/// from a table EOS.
/// @details `update` solves the RT for the current rho and e, and stores
/// Q_rad [erg cm^-3 s^-1] in `qrad` on the Backend. The intensity of the
/// previous call is the initial guess of the iteration.
///
/// The z axis is the vertical direction (upward) and must not be periodic. The
/// x and y directions must be periodic unless their size is 1 (the incoming
/// intensity is set only at the z boundaries). A 2D domain in x-z
/// (j_size = 1) is allowed.
///
/// S and alpha are computed in all cells, including the ghost cells, and the
/// incoming intensity at the bottom is S of the lowest ghost cell used by RT.
/// So the ghost cells must hold valid states in the table (the MHD boundary
/// condition sets them), and the bottom ghost cell a thermalized deep state.
///
/// The settings `rt.num_rays` (must be 24), `rt.tolerance` and `rt.max_iters`
/// are read from the config.
///
template <typename Real, typename Backend> struct Radiation {
  RT<Real> rt;
  RadiativeHeating<Real> heating;
  /// @brief Q_rad on the Backend (0 in the ghost cells)
  Array3D<Real, Backend> qrad;

  Radiation(const Config &config, const Grid<Real> &grid)
      : rt(grid, config["rt"]["num_rays"].template as<int>()), heating(grid),
        qrad(grid.i_total, grid.j_total, grid.k_total),
        ro_h(grid.i_total, grid.j_total, grid.k_total),
        ei_h(grid.i_total, grid.j_total, grid.k_total),
        tolerance(config["rt"]["tolerance"].template as<Real>()),
        max_iters(config["rt"]["max_iters"].template as<int>()) {
    const auto periodic = config["domain"]["periodic"];
    if ((grid.i_size > 1 && !periodic["x"].template as<bool>()) ||
        (grid.j_size > 1 && !periodic["y"].template as<bool>())) {
      throw std::runtime_error("Radiation needs periodic x and y directions");
    }
    if (grid.k_size == 1 || periodic["z"].template as<bool>()) {
      throw std::runtime_error("Radiation needs a non-periodic z direction");
    }
    if (rt.ang_quad.num_rays != 24) {
      throw std::runtime_error("Radiation needs 24 rays");
    }
    qrad.copy_from(heating.qrad);  // zero
  }

  /// @brief Solve the RT for rho and e of `qq` and set `qrad`.
  void update(const mhd::Fields<Real, Backend> &qq,
              const eos::TableEOS<Real, Backend> &eos, const Grid<Real> &grid,
              const mpi::Shape &mpi_shape) {
    ro_h.copy_from(qq.ro);
    ei_h.copy_from(qq.ei);
    set_source_and_absorption(rt, grid, ro_h, ei_h, eos);
    rt.solve(grid, mpi_shape, tolerance, max_iters, VerticalBoundary{});
    heating.compute(rt, grid, mpi_shape);
    qrad.copy_from(heating.qrad);
  }

  /// @brief Source term that adds `qrad` to the energy equation
  RadiativeHeatingSource<Real> source() const { return {qrad.const_view()}; }

private:
  Array3D<Real> ro_h, ei_h;
  Real tolerance;
  int max_iters;
};

}  // namespace rt
}  // namespace miso
