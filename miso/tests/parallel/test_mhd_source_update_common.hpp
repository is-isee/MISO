#pragma once

#include <string>

#include <doctest/doctest.h>

#include <miso/core.hpp>
#include <miso/mhd.hpp>

// Integration test of the source terms through the MHD update.
//
// The initial state is uniform (v = 0, B = 0) with periodic boundaries, so
// that all flux terms and the artificial viscosity vanish. With constant
// sources S, one Runge-Kutta step then gives exactly q = q0 + dt * S for the
// conserved variables, which checks that each source term is added to the
// right equation and scaled by the time step.

namespace test_source_update {

using Real = double;
using namespace miso;

constexpr Real ro0 = 1.0;
constexpr Real pr0 = 1.0;

/// @brief Constant sources: mass, momentum, induction, energy
struct Rates {
  Real ro, vx, vy, vz, bx, by, bz, ei;
};
constexpr Rates heating_only{0, 0, 0, 0, 0, 0, 0, 2.5};
constexpr Rates all_terms{0.3, 0.2, -0.4, 0.5, 0.7, -0.6, 0.8, 2.5};

struct UniformState {
  eos::IdealEOS<Real> &eos;
  void apply(mhd::FieldsView<Real> qq, GridView<const Real>) const {
    for (int n = 0; n < qq.size(); ++n) {
      qq.ro[n] = ro0;
      qq.vx[n] = qq.vy[n] = qq.vz[n] = 0.0;
      qq.bx[n] = qq.by[n] = qq.bz[n] = 0.0;
      qq.ei[n] = pr0 / (eos.gm - 1.0) / ro0;
      qq.ph[n] = 0.0;
    }
  }
};

/// @brief Defines only the heating term (the others are omitted)
struct HeatingSource {
  __host__ __device__ Real ei(mhd::FieldsView<const Real>, int, int, int) const {
    return heating_only.ei;
  }
};

/// @brief Defines every term, and a time step limit that differs by rank
struct FullSource {
  // clang-format off
  __host__ __device__ Real ro(mhd::FieldsView<const Real>, int, int, int) const { return all_terms.ro; }
  __host__ __device__ Real vx(mhd::FieldsView<const Real>, int, int, int) const { return all_terms.vx; }
  __host__ __device__ Real vy(mhd::FieldsView<const Real>, int, int, int) const { return all_terms.vy; }
  __host__ __device__ Real vz(mhd::FieldsView<const Real>, int, int, int) const { return all_terms.vz; }
  __host__ __device__ Real bx(mhd::FieldsView<const Real>, int, int, int) const { return all_terms.bx; }
  __host__ __device__ Real by(mhd::FieldsView<const Real>, int, int, int) const { return all_terms.by; }
  __host__ __device__ Real bz(mhd::FieldsView<const Real>, int, int, int) const { return all_terms.bz; }
  __host__ __device__ Real ei(mhd::FieldsView<const Real>, int, int, int) const { return all_terms.ei; }
  // clang-format on

  /// @brief 1e-3 on rank 0, 2e-3 on rank 1, ...
  Real dt_limit() const {
    int rank;
    MPI_Comm_rank(mpi::comm(), &rank);
    return 1.e-3 * (rank + 1);
  }
};

template <typename Backend, typename Source>
struct Model : public mhd::ModelBase<Model<Backend, Source>, Real, Backend> {
  eos::IdealEOS<Real> eos;
  UniformState ic;
  mhd::EmptyBoundaryCondition<Real> bc;
  Source src;

  explicit Model(Config &config)
      : mhd::ModelBase<Model<Backend, Source>, Real, Backend>(config),
        eos(config), ic{eos}, bc(), src() {}
};

/// @brief Check the state after one step of dt with the constant sources s
template <typename ModelType>
void check_state(const ModelType &model, Real dt, const Rates &s) {
  const auto &g = model.grid;
  mhd::Fields<Real, backend::Host> qq(g);
  qq.copy_from(model.mhd.qq);

  const Real ro = ro0 + dt * s.ro;
  const Real vx = dt * s.vx / ro, vy = dt * s.vy / ro, vz = dt * s.vz / ro;
  const Real bx = dt * s.bx, by = dt * s.by, bz = dt * s.bz;
  const Real e_total = pr0 / (model.eos.gm - 1.0) + dt * s.ei;
  const Real ei = (e_total - 0.5 * ro * (vx * vx + vy * vy + vz * vz) -
                   pii8<Real> * (bx * bx + by * by + bz * bz)) /
                  ro;
  const auto approx = [](Real v) {
    return doctest::Approx(v).epsilon(1e-12).scale(1.0);
  };
  for (int i = g.i_margin; i < g.i_total - g.i_margin; ++i) {
    for (int j = g.j_margin; j < g.j_total - g.j_margin; ++j) {
      for (int k = g.k_margin; k < g.k_total - g.k_margin; ++k) {
        CHECK(qq.ro(i, j, k) == approx(ro));
        CHECK(qq.vx(i, j, k) == approx(vx));
        CHECK(qq.vy(i, j, k) == approx(vy));
        CHECK(qq.vz(i, j, k) == approx(vz));
        CHECK(qq.bx(i, j, k) == approx(bx));
        CHECK(qq.by(i, j, k) == approx(by));
        CHECK(qq.bz(i, j, k) == approx(bz));
        CHECK(qq.ei(i, j, k) == approx(ei));
      }
    }
  }
}

inline std::string config_path() {
  return std::string(CONFIG_DIR) + "config_source.yaml";
}

/// @brief Advance one step with a given dt (heating only)
template <typename Backend> void check_heating(Real dt) {
  Config config(config_path());
  Model<Backend, HeatingSource> model(config);
  model.mhd.apply_initial_condition(model.ic, model.bc);
  model.mhd.update(dt, model.eos, model.bc, model.src);
  check_state(model, dt, heating_only);
}

/// @brief Advance one step with a given dt (all terms)
template <typename Backend> void check_all_terms(Real dt) {
  Config config(config_path());
  Model<Backend, FullSource> model(config);
  model.mhd.apply_initial_condition(model.ic, model.bc);
  model.mhd.update(dt, model.eos, model.bc, model.src);
  check_state(model, dt, all_terms);
}

/// @brief ModelBase::update uses the minimum of dt_limit() over all ranks
template <typename Backend> void check_dt_limit() {
  Config config(config_path());
  Model<Backend, FullSource> model(config);
  model.mhd.apply_initial_condition(model.ic, model.bc);
  model.update();
  CHECK(model.time.time == doctest::Approx(1.e-3).epsilon(1e-14));
  check_state(model, 1.e-3, all_terms);
}

}  // namespace test_source_update
