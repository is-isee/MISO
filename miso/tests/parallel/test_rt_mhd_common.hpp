#pragma once

#include <cmath>
#include <string>

#include <doctest/doctest.h>

#include <miso/core.hpp>
#include <miso/eos_table.hpp>
#include <miso/mhd.hpp>
#include <miso/rt.hpp>
#include <miso/rt_mhd.hpp>

// Integration test of rt::Radiation (RT on the host) with the MHD solver.
//
// The state is uniform (v = 0, B = 0) at a table point in the photosphere, in
// a vertical 2D domain (x-z, j_size = 1) split in z over two ranks. The
// optical depth of the domain is about 3. Q_rad is then negative (cooling)
// near the top. One MHD step changes the energy of the domain by dt times the
// sum of Q_rad, with Q_rad fixed during the step.

namespace test_rt_mhd {

using Real = double;
using namespace miso;

// A table point in input_data/eos_table_roen_blend.tbl (see test_eos_table.cpp)
const Real ro0 = std::exp(-15.377656523015137);
const Real ei0 = std::exp(27.981135057266535);

struct UniformState {
  void apply(mhd::FieldsView<Real> qq, GridView<const Real>) const {
    for (int n = 0; n < qq.size(); ++n) {
      qq.ro[n] = ro0;
      qq.vx[n] = qq.vy[n] = qq.vz[n] = 0.0;
      qq.bx[n] = qq.by[n] = qq.bz[n] = 0.0;
      qq.ei[n] = ei0;
      qq.ph[n] = 0.0;
    }
  }
};

/// @brief Mirror at the z boundaries (walls), so that the uniform state stays
/// uniform there.
template <typename Backend> struct WallBoundary {
  const mpi::Shape &mpi_shape;
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {
    namespace bc = miso::boundary_condition;
    for (const Side side : {Side::INNER, Side::OUTER}) {
      if (!bc::is_physical_boundary(Direction::Z, side, mpi_shape)) {
        continue;
      }
      for (auto arr : {qq.ro, qq.vx, qq.vy, qq.bx, qq.by, qq.bz, qq.ei, qq.ph}) {
        bc::symmetric(Backend{}, arr, grid, Sign::Pos, Direction::Z, side);
      }
      bc::symmetric(Backend{}, qq.vz, grid, Sign::Neg, Direction::Z, side);
    }
  }
};

template <typename Backend>
struct Model : public mhd::ModelBase<Model<Backend>, Real, Backend> {
  using Base = mhd::ModelBase<Model<Backend>, Real, Backend>;
  eos::TableEOS<Real, Backend> eos;
  UniformState ic;
  WallBoundary<Backend> bc;
  rt::Radiation<Real, Backend> radiation;
  rt::RadiativeHeatingSource<Real> src;

  explicit Model(Config &config)
      : Base(config), eos(fs::path(INPUT_DATA_DIR) / "eos_table_roen_blend.tbl"),
        bc{this->mpi_shape}, radiation(config, this->grid),
        src(radiation.source()) {}
};

template <typename Backend> void check_radiation() {
  Config config(std::string(CONFIG_DIR) + "config_rt_mhd.yaml");
  Model<Backend> model(config);
  model.mhd.apply_initial_condition(model.ic, model.bc);
  const auto &g = model.grid;
  REQUIRE(g.j_size == 1);
  REQUIRE(model.mpi_shape.z_procs == 2);

  model.radiation.update(model.mhd.qq, model.eos, g, model.mpi_shape);
  const auto &heating = model.radiation.heating;
  Array3D<Real> qrad(g.i_total, g.j_total, g.k_total);
  qrad.copy_from(model.radiation.qrad);

  // Topmost physical cell of the whole domain (only on the upper rank)
  const bool top_rank = model.mpi_shape.z_procs_pos == MPI_PROC_NULL;
  const int k_top = g.k_margin + g.k_size - 1;
  for (int i = g.i_margin; i < g.i_margin + g.i_size; ++i) {
    for (int k = g.k_margin; k < g.k_margin + g.k_size; ++k) {
      // Copied to the Backend as it is, and uniform in x
      CHECK(qrad(i, 0, k) == heating.qrad(i, 0, k));
      CHECK(qrad(i, 0, k) ==
            doctest::Approx(qrad(g.i_margin, 0, k)).epsilon(1e-8));
      if (top_rank && k == k_top) {
        CHECK(qrad(i, 0, k) == 0);  // tau = 0 < tau_gate
      } else {
        CHECK(qrad(i, 0, k) < 0);  // cooling, J < S
      }
    }
  }
  if (top_rank) {
    // Cooling is the strongest near the top
    const int i = g.i_margin;
    CHECK(qrad(i, 0, k_top - 1) < qrad(i, 0, k_top - 4));
  }
  {
    // Weak cooling at the bottom (tau ~ 3), where the incoming intensity is
    // S: about 1.4% of that near the top.
    Real q_top = top_rank ? qrad(g.i_margin, 0, k_top - 1) : Real(0);
    Real q_top_g;
    MPI_Allreduce(&q_top, &q_top_g, 1, mpi::data_type<Real>(), MPI_MIN,
                  model.mpi_shape.cart_comm);
    if (model.mpi_shape.z_procs_neg == MPI_PROC_NULL) {
      CHECK(std::abs(qrad(g.i_margin, 0, g.k_margin)) < 0.05 * std::abs(q_top_g));
    }
  }

  // One MHD step with the source: the energy of the domain changes by
  // dt * (sum of Q_rad), as the walls and the periodic x boundaries let no
  // energy through. (In each cell, e changes by dt Q_rad / rho only roughly,
  // because the artificial diffusion and the pressure gradient move the energy
  // within the step.)
  const Real dt = model.mhd.cfl(model.eos);
  model.mhd.update(dt, model.eos, model.bc, model.src);
  mhd::Fields<Real, backend::Host> qq(g);
  qq.copy_from(model.mhd.qq);
  Real sums[2] = {0, 0};  // change of the energy, dt * Q_rad
  for (int i = g.i_margin; i < g.i_margin + g.i_size; ++i) {
    for (int k = g.k_margin; k < g.k_margin + g.k_size; ++k) {
      const Real v2 = util::pow2(qq.vx(i, 0, k)) + util::pow2(qq.vz(i, 0, k));
      sums[0] += qq.ro(i, 0, k) * (qq.ei(i, 0, k) + Real(0.5) * v2) - ro0 * ei0;
      sums[1] += dt * qrad(i, 0, k);
    }
  }
  Real sums_g[2];
  MPI_Allreduce(sums, sums_g, 2, mpi::data_type<Real>(), MPI_SUM,
                model.mpi_shape.cart_comm);
  CHECK(sums_g[1] < 0);
  CHECK(sums_g[0] == doctest::Approx(sums_g[1]).epsilon(1e-8));
}

}  // namespace test_rt_mhd
