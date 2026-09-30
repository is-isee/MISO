#pragma once

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "config.hpp"
#include "env.hpp"
#include "eos.hpp"
#include "execution.hpp"
#include "grid.hpp"
#include "mhd_solver.hpp"
#include "mpi_util.hpp"
#include "time.hpp"
#include "types.hpp"

namespace miso {
namespace mhd {

// SFINAE to check if the derived class has required member variables.
template <class T, class = void> struct has_eos : std::false_type {};
template <class T>
struct has_eos<T, std::void_t<decltype(&T::eos)>> : std::true_type {};

template <class T, class = void> struct has_ic : std::false_type {};
template <class T>
struct has_ic<T, std::void_t<decltype(&T::ic)>> : std::true_type {};

template <class T, class = void> struct has_bc : std::false_type {};
template <class T>
struct has_bc<T, std::void_t<decltype(&T::bc)>> : std::true_type {};

template <class T, class = void> struct has_src : std::false_type {};
template <class T>
struct has_src<T, std::void_t<decltype(&T::src)>> : std::true_type {};

// Optional time step limit of the source term: Real dt_limit() const
template <class T, class = void> struct has_dt_limit : std::false_type {};
template <class T>
struct has_dt_limit<T,
                    std::void_t<decltype(std::declval<const T &>().dt_limit())>>
    : std::true_type {};
template <class T, class = void> struct has_member_dt_limit : std::false_type {};
template <class T>
struct has_member_dt_limit<T, std::void_t<decltype(&T::dt_limit)>>
    : std::true_type {};

/// @brief Base class of MHD models using CRTP.
/// @details The derived class must implement the following members:
/// - eos: equation of state
/// - ic: initial condition
/// - bc: boundary condition
/// - src: source term. Every term (ro, vx, vy, vz, bx, by, bz, ei) and
///   dt_limit(), the upper limit of the time step, are optional
///   (EmptySourceTerm defines none of them).
template <class Derived, class Real, class Backend> class ModelBase {
public:
  Config &config;
  mpi::Shape mpi_shape;
  Time<Real> time;
  Grid<Real, backend::Host> grid;

  ExecContext<Real, Backend> exec_ctx;
  MHD<Real, Backend> mhd;

  explicit ModelBase(Config &cfg)
      : config(cfg), mpi_shape(cfg), time(cfg), grid(cfg, mpi_shape),
        exec_ctx(mpi_shape, grid), mhd(cfg, grid, exec_ctx) {}

  /// @brief Default implementation of one timestep update.
  /// @details The derived class may provide its own `update()` method.
  void update() {
    auto &d = derived();
    using Source = decltype(d.src);
    static_assert(!has_member_dt_limit<Source>::value ||
                      has_dt_limit<Source>::value,
                  "Source::dt_limit must be callable as dt_limit() const");
    auto dt = mhd.cfl(d.eos);
    if constexpr (has_dt_limit<Source>::value) {
      // dt_limit() returns the limit of this rank: take the global minimum.
      // A non-positive (or NaN) value means that the source term is in an
      // invalid state; it is reported on every rank, so that no rank is left
      // waiting in a collective communication.
      Real dt_src = static_cast<Real>(d.src.dt_limit());
      if (!(dt_src > Real(0))) {
        dt_src = Real(-1);
      }
      Real dt_src_g;
      MPI_Allreduce(&dt_src, &dt_src_g, 1, mpi::data_type<Real>(), MPI_MIN,
                    mpi::comm());
      if (dt_src_g <= Real(0)) {
        throw std::runtime_error(
            "ModelBase::update: dt_limit() of the source term is not "
            "positive on some rank.");
      }
      dt = std::min(dt, dt_src_g);
    }
    mhd.update(dt, d.eos, d.bc, d.src);
    time.update(dt);
  }

  void save_metadata() {
    MPI_Barrier(mpi::comm());
    config.save();
    grid.save(config);
    exec_ctx.mpi_shape.save();
  }

  void save_state() {
    time.save();
    mhd.save(time);
  }

  void load_state() {
    time.load();
    mhd.load(time);
  }

  void save_if_needed() {
    if (time.time >= time.dt_output * time.n_output) {
      save_state();
      time.log();
      time.n_output++;
    }
  }

  void run() {
    // Ensure to call methods in the derived class
    auto &d = derived();

    // Assert that d has eos, ic, bc, and src as member variables.
    static_assert(has_eos<Derived>::value,
                  "Derived must have member variable 'eos'.");
    static_assert(has_ic<Derived>::value,
                  "Derived must have member variable 'ic'.");
    static_assert(has_bc<Derived>::value,
                  "Derived must have member variable 'bc'.");
    static_assert(has_src<Derived>::value,
                  "Derived must have member variable 'src'.");

    mhd.apply_initial_condition(d.ic, d.bc);
    if (config["io"]["continue"].template as<bool>() &&
        fs::exists(time.n_output_filepath())) {
      load_state();
    }

    save_metadata();
    save_if_needed();

    while (time.time < time.tend) {
      d.update();
      save_if_needed();
    }
  }

protected:
  Derived &derived() { return static_cast<Derived &>(*this); }
};

/// @brief Empty boundary condition class (e.g., periodic in all directions).
/// @details Periodic boundary condition is applied by MPI communication.
/// Be sure to set "periodic" in domain field of config.yaml.
template <typename Real> struct EmptyBoundaryCondition {
  void apply(mhd::FieldsView<Real>, GridView<const Real>) const {}
};

/// @brief Empty source term class (without source terms).
/// @details All source terms are optional (see mhd_integrator.hpp): a source
/// term class defines only the terms it needs, e.g.,
/// @code
/// __host__ __device__ Real vx(FieldsView<const Real> qq, int i, int j,
///                             int k) const;  // external force in x
/// Real dt_limit() const;  // optional upper limit of the time step
/// @endcode
/// The terms are called in CUDA kernels with the CUDA backend, so they must
/// be `__host__ __device__` and use only device-accessible data (e.g., views
/// of arrays of the backend). dt_limit() is called on the host every step and
/// returns the limit of the rank; the minimum over all MPI ranks is used. A
/// non-positive value on any rank throws std::runtime_error on every rank.
template <typename Real> struct EmptySourceTerm {};

}  // namespace mhd
}  // namespace miso
