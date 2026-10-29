// Solar photosphere in 3D: convection with the table EOS, gravity, and
// radiative heating and cooling by the RT with 24 rays (#201).
//
// - Initial condition: the 1D stratification in input_data/ and a small random
//   vertical velocity.
// - Bottom: open. Upflows come in with a fixed entropy (that of the initial
//   stratification at the bottom), and downflows go out as they are.
// - Top: closed wall (vz = 0), with an absorbing layer below it.
// - The velocity is damped at the start, to calm the transient of the initial
//   state (it is not in radiative equilibrium).
// - The RT is solved on the host every step (rt::Radiation).
// The boundaries and the damping follow the photospheric runs of R2D2plus.

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <miso/core.hpp>
#include <miso/eos_table.hpp>
#include <miso/mhd.hpp>
#include <miso/rt_mhd.hpp>
#include <miso/stratification.hpp>

using namespace miso;

using Real = double;

#ifdef USE_CUDA
using Backend = backend::CUDA;
#else
using Backend = backend::Host;
#endif

struct InitialCondition {
  const Stratification<Real> &strat;
  Real v_amp;

  // The signature must not be changed as it is called inside miso::mhd::MHD.
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {
    std::mt19937 engine(mpi::rank());
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);
    for (int k = 0; k < grid.k_total; ++k) {
      const auto s = strat.at(grid.z[k]);
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          qq.ro(i, j, k) = s.ro;
          qq.ei(i, j, k) = s.ei;
          qq.vx(i, j, k) = 0.0;
          qq.vy(i, j, k) = 0.0;
          qq.vz(i, j, k) = v_amp * dist(engine);
          qq.bx(i, j, k) = 0.0;
          qq.by(i, j, k) = 0.0;
          qq.bz(i, j, k) = 0.0;
          qq.ph(i, j, k) = 0.0;
        }
      }
    }
  }
};

/// @brief ln e at a fixed entropy as a function of ln rho, on the rho axis of
/// the EOS table (for the upflows at the bottom).
/// @details In each row of the table (fixed rho), e is found where the
/// entropy crosses `se0`, linearly in ln e.
inline Array1D<Real> log_ei_at_entropy(const eos::TableFile &file, Real se0) {
  const int n_ro = file.shapes.at("ro")[0];
  const int n_en = file.shapes.at("en")[0];
  const auto &en = file.get("en", {n_en});
  const auto &se = file.get("se", {n_ro, n_en});
  const auto &valid = file.get("valid", {n_ro, n_en});
  Array1D<Real> out(n_ro);
  for (int i = 0; i < n_ro; ++i) {
    out[i] = std::numeric_limits<Real>::quiet_NaN();
    for (int j = 0; j + 1 < n_en; ++j) {
      const int n = i * n_en + j;
      if (valid[n] == 0 || valid[n + 1] == 0) {
        continue;
      }
      if (se[n] <= se0 && se0 < se[n + 1]) {
        out[i] =
            en[j] + (se0 - se[n]) / (se[n + 1] - se[n]) * (en[j + 1] - en[j]);
        break;
      }
    }
  }
  return out;
}

/// @brief Boundaries in z. A ghost cell g takes the values of its mirror cell
/// m across the wall (the first ghost layer mirrors the first physical layer,
/// and so on), as in the photospheric boundaries of R2D2:
/// - rho_g = rho_m rho0_g / rho0_m and e_g = e_m e0_g / e0_m, with rho0 and e0
///   of the initial stratification, except e_g in the upflows at the bottom,
///   which is e at the entropy of the initial stratification at the bottom
/// - v_g = f v_m with f = rho_m / rho_g, so that the mass flux rho v is
///   mirrored (the sign of vz is flipped at the top, a closed wall)
/// - At the bottom, the horizontal mean of the relative density perturbation
///   rho / rho0 - 1 is flipped in sign (the deviations from the mean are kept),
///   so that mass piling up at the bottom is pushed out, and vice versa.
struct BoundaryCondition {
  const mpi::Shape &mpi_shape;
  /// @brief ln e of the upflows at the bottom on the ln rho axis of the table
  Array1D<Real, Backend> log_ei_upflow;
  Real log_ro_min, dlog_ro_inv;
  /// @brief rho0_g / rho0_m and e0_g / e0_m for the ghost layers 1, 2, ... at
  /// the bottom, then at the top
  Array1D<Real, Backend> ro_ratio, ei_ratio;
  /// @brief rho0 of the ghost layers and their mirror layers at the bottom
  std::vector<Real> ro0_ghost, ro0_mirror;
  /// @brief 2 <rho / rho0 - 1>_m rho0_g for the ghost layers at the bottom
  mutable Array1D<Real> ro_shift_h;
  mutable Array1D<Real, Backend> ro_shift;

  BoundaryCondition(const mpi::Shape &mpi_shape, const Grid<Real> &grid,
                    const Stratification<Real> &strat, const eos::TableFile &file,
                    Real se0)
      : mpi_shape(mpi_shape), log_ei_upflow(file.shapes.at("ro")[0]),
        ro_ratio(2 * grid.k_margin), ei_ratio(2 * grid.k_margin),
        ro_shift_h(grid.k_margin), ro_shift(grid.k_margin) {
    const auto &ro = file.get("ro", {log_ei_upflow.size()});
    log_ro_min = ro.front();
    dlog_ro_inv = (ro.size() - 1) / (ro.back() - ro.front());
    const auto le = log_ei_at_entropy(file, se0);
    log_ei_upflow.copy_from(le);
    // ln rho of the rows used at the bottom (within a factor e of the initial
    // bottom ghost cells) must have the entropy se0
    const int km = grid.k_margin;
    for (int i = 0; i < le.size(); ++i) {
      const Real lr = ro[0] + i / dlog_ro_inv;
      if (std::log(strat.at(grid.z[0]).ro) - 1 < lr &&
          lr < std::log(strat.at(grid.z[km]).ro) + 1 && !std::isfinite(le[i])) {
        throw std::runtime_error(
            "The EOS table has no e at the bottom entropy for ln rho = " +
            std::to_string(lr));
      }
    }

    const int k0 = grid.k_total - km - 1;  // topmost physical cell
    Array1D<Real> ro_r(2 * km), ei_r(2 * km);
    for (int m = 1; m <= km; ++m) {
      for (const auto [n, k_g, k_m] :
           {std::array<int, 3>{m - 1, km - m, km + m - 1},
            std::array<int, 3>{km + m - 1, k0 + m, k0 + 1 - m}}) {
        const auto s_g = strat.at(grid.z[k_g]);
        const auto s_m = strat.at(grid.z[k_m]);
        ro_r[n] = s_g.ro / s_m.ro;
        ei_r[n] = s_g.ei / s_m.ei;
      }
    }
    ro_ratio.copy_from(ro_r);
    ei_ratio.copy_from(ei_r);
    for (int m = 1; m <= km; ++m) {
      ro0_ghost.push_back(strat.at(grid.z[km - m]).ro);
      ro0_mirror.push_back(strat.at(grid.z[km + m - 1]).ro);
    }
  }

  /// @brief Set ro_shift from the mean density perturbation of the mirror
  /// layers at the bottom (collective: called on all ranks)
  void set_ro_shift(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {
    const int km = grid.k_margin;
    std::vector<Real> sums(km + 1, Real(0));  // per layer, then the count
    if (boundary_condition::is_physical_boundary(Direction::Z, Side::INNER,
                                                 mpi_shape)) {
      for (int m = 1; m <= km; ++m) {
        const int k_m = km + m - 1;
        const Real ro0_inv = Real(1) / ro0_mirror[m - 1];
        const Range3D range{{grid.i_margin, grid.i_total - grid.i_margin},
                            {grid.j_margin, grid.j_total - grid.j_margin},
                            {k_m, k_m + 1}};
        sums[m - 1] = reduce(
            Backend{}, range, Real(0),
            MISO_LAMBDA(int i, int j, int k) {
              return qq.ro(i, j, k) * ro0_inv - Real(1);
            },
            MISO_LAMBDA(Real a, Real b) { return a + b; });
      }
      sums[km] = Real((grid.i_total - 2 * grid.i_margin) *
                      (grid.j_total - 2 * grid.j_margin));
    }
    MPI_Allreduce(MPI_IN_PLACE, sums.data(), km + 1, mpi::data_type<Real>(),
                  MPI_SUM, mpi::comm());
    for (int m = 1; m <= km; ++m) {
      ro_shift_h[m - 1] = Real(2) * sums[m - 1] / sums[km] * ro0_ghost[m - 1];
    }
    ro_shift.copy_from(ro_shift_h);
  }

  // The signature must not be changed as it is called by miso integrator.
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {
    namespace bc = miso::boundary_condition;
    const Backend btag{};
    for (const Side side : {Side::INNER, Side::OUTER}) {
      if (!bc::is_physical_boundary(Direction::Z, side, mpi_shape)) {
        continue;
      }
      for (auto arr : {qq.bx, qq.by, qq.bz, qq.ph}) {
        bc::symmetric(btag, arr, grid, Sign::Pos, Direction::Z, side);
      }
    }

    // device lambdas must not capture `this`
    const int km = grid.k_margin;
    const auto ro_r = ro_ratio.const_view();
    const auto ei_r = ei_ratio.const_view();
    set_ro_shift(qq, grid);
    const auto shift = ro_shift.const_view();
    if (bc::is_physical_boundary(Direction::Z, Side::INNER, mpi_shape)) {
      const auto le = log_ei_upflow.const_view();
      const int n_ro = le.size();
      const Real lr_min = log_ro_min, dlr_inv = dlog_ro_inv;
      const Range3D range{{0, grid.i_total}, {0, grid.j_total}, {0, km}};
      for_each(
          btag, range, MISO_LAMBDA(int i, int j, int k) {
            const int m = km - k;  // ghost layer 1, 2, ...
            const int k_m = km + m - 1;
            const Real ro = qq.ro(i, j, k_m) * ro_r[m - 1] - shift[m - 1];
            const Real f = qq.ro(i, j, k_m) / ro;
            qq.ro(i, j, k) = ro;
            qq.vx(i, j, k) = f * qq.vx(i, j, k_m);
            qq.vy(i, j, k) = f * qq.vy(i, j, k_m);
            qq.vz(i, j, k) = f * qq.vz(i, j, k_m);
            if (qq.vz(i, j, k_m) > Real(0)) {
              // upflow: e at the entropy of the bottom
              Real w = util::clamp<Real>((util::log(ro) - lr_min) * dlr_inv,
                                         Real(0), Real(n_ro - 1));
              const int n = util::clamp<int>(static_cast<int>(w), 0, n_ro - 2);
              w -= n;
              qq.ei(i, j, k) = util::exp((Real(1) - w) * le[n] + w * le[n + 1]);
            } else {
              qq.ei(i, j, k) = qq.ei(i, j, k_m) * ei_r[m - 1];
            }
          });
    }
    if (bc::is_physical_boundary(Direction::Z, Side::OUTER, mpi_shape)) {
      const int k0 = grid.k_total - km - 1;  // topmost physical cell
      const Range3D range{
          {0, grid.i_total}, {0, grid.j_total}, {k0 + 1, grid.k_total}};
      for_each(
          btag, range, MISO_LAMBDA(int i, int j, int k) {
            const int m = k - k0;  // ghost layer 1, 2, ...
            const int k_m = k0 + 1 - m;
            const Real ro = qq.ro(i, j, k_m) * ro_r[km + m - 1];
            const Real f = qq.ro(i, j, k_m) / ro;
            qq.ro(i, j, k) = ro;
            qq.vx(i, j, k) = f * qq.vx(i, j, k_m);
            qq.vy(i, j, k) = f * qq.vy(i, j, k_m);
            qq.vz(i, j, k) = -f * qq.vz(i, j, k_m);
            qq.ei(i, j, k) = qq.ei(i, j, k_m) * ei_r[km + m - 1];
          });
    }
  }
};

struct SourceTerm {
  rt::RadiativeHeatingSource<Real> rad;
  /// @brief Gravitational acceleration [cm/s^2] toward -z, as a function of k
  Array1DView<const Real> gx;
  /// @brief Rate [1/s] of the velocity damping at the start (set every step)
  Real damping_init;
  /// @brief Rate [1/s] of the velocity damping near the top, as a function of
  /// k (absorbing layer), and its maximum
  Array1DView<const Real> damping_top;
  Real damping_top_max;

  /// @brief The damping is explicit: rate * dt <= 0.5
  Real dt_limit() const { return Real(0.5) / (damping_init + damping_top_max); }

  __host__ __device__ Real damping(int k) const {
    return damping_init + damping_top[k];
  }

  __host__ __device__ Real vx(mhd::FieldsView<const Real> qq, int i, int j,
                              int k) const {
    return -damping(k) * qq.ro(i, j, k) * qq.vx(i, j, k);
  }

  __host__ __device__ Real vy(mhd::FieldsView<const Real> qq, int i, int j,
                              int k) const {
    return -damping(k) * qq.ro(i, j, k) * qq.vy(i, j, k);
  }

  __host__ __device__ Real vz(mhd::FieldsView<const Real> qq, int i, int j,
                              int k) const {
    return -(gx[k] + damping(k) * qq.vz(i, j, k)) * qq.ro(i, j, k);
  }

  /// Q_rad and the work of the forces
  __host__ __device__ Real ei(mhd::FieldsView<const Real> qq, int i, int j,
                              int k) const {
    return rad.ei(qq, i, j, k) + vx(qq, i, j, k) * qq.vx(i, j, k) +
           vy(qq, i, j, k) * qq.vy(i, j, k) + vz(qq, i, j, k) * qq.vz(i, j, k);
  }
};

struct Model : public mhd::ModelBase<Model, Real, Backend> {
  fs::path input_dir;
  eos::TableFile eos_file;
  eos::TableEOS<Real, Backend> eos;
  Stratification<Real> strat;
  InitialCondition ic;
  BoundaryCondition bc;
  rt::Radiation<Real, Backend> radiation;
  Array1D<Real, Backend> gx, damping_top;
  SourceTerm src;

  explicit Model(Config &config)
      : ModelBase(config),
        input_dir(fs::absolute(config.load_filepath).parent_path() /
                  config["photosphere"]["input_data"].as<std::string>()),
        eos_file(input_dir / "eos_table_roen_blend.tbl"), eos(eos_file),
        strat(input_dir / "stratification_miso.tbl"),
        ic{strat, config["photosphere"]["v_amp"].as<Real>()},
        bc(mpi_shape, grid, strat, eos_file, bottom_entropy()),
        radiation(config, grid), gx(gravity()),
        damping_top(
            top_layer(config["photosphere"]["top_layer"].as<Real>(),
                      config["photosphere"]["top_layer_rate"].as<Real>())),
        src{radiation.source(), gx.const_view(), Real(0),
            damping_top.const_view(),
            config["photosphere"]["top_layer_rate"].as<Real>()},
        damping_end(config["photosphere"]["damping_end"].as<Real>()),
        damping_tau(config["photosphere"]["damping_tau"].as<Real>()) {}

  /// @brief RT for the current state, then one MHD step
  void update() {
    radiation.update(mhd.qq, eos, grid, mpi_shape);
    // once after each output (the state of the output)
    if (time.n_output - 1 > n_teff) {
      save_teff();
      n_teff = time.n_output - 1;
    }
    // The damping weakens linearly and vanishes at damping_end, as
    // [physics.initial_damping] of R2D2plus
    src.damping_init = time.time < damping_end
                           ? (damping_end - time.time) / damping_end / damping_tau
                           : Real(0);
    ModelBase::update();
  }

private:
  /// @brief gx of the stratification at the heights of the cells
  Array1D<Real, Backend> gravity() const {
    Array1D<Real> g(grid.k_total);
    for (int k = 0; k < grid.k_total; ++k) {
      g[k] = strat.at(grid.z[k]).gx;
    }
    Array1D<Real, Backend> out(grid.k_total);
    out.copy_from(g);
    return out;
  }

  /// @brief Damping rate rate ((z - z_edge) / thickness)^2 above z_edge, the
  /// thickness below the top (as top_velocity_damping of R2D2plus)
  Array1D<Real, Backend> top_layer(Real thickness, Real rate) const {
    const Real z_edge = config["grid"]["z_max"].as<Real>() - thickness;
    Array1D<Real> w(grid.k_total);
    for (int k = 0; k < grid.k_total; ++k) {
      w[k] = grid.z[k] > z_edge
                 ? rate * util::pow2((grid.z[k] - z_edge) / thickness)
                 : Real(0);
    }
    Array1D<Real, Backend> out(grid.k_total);
    out.copy_from(w);
    return out;
  }

  /// @brief End [s] and initial time scale [s] of the velocity damping
  Real damping_end, damping_tau;
  int n_teff = -1;

  /// @brief Entropy of the table at the initial state at the bottom
  Real bottom_entropy() const {
    const auto s = strat.at(grid.z[grid.k_margin]);
    const int n_ro = eos_file.shapes.at("ro")[0];
    const int n_en = eos_file.shapes.at("en")[0];
    const auto &ro = eos_file.get("ro", {n_ro});
    const auto &en = eos_file.get("en", {n_en});
    const std::vector<Real> se(eos_file.get("se", {n_ro, n_en}).begin(),
                               eos_file.get("se", {n_ro, n_en}).end());
    return interpolate_uniform_table2d(
        se.data(), Real(ro.front()), Real((n_ro - 1) / (ro.back() - ro.front())),
        Real(en.front()), Real((n_en - 1) / (en.back() - en.front())),
        std::log(s.ro), std::log(s.ei), n_ro, n_en);
  }

  /// @brief Append the time and T_eff (from the mean vertical flux at the
  /// topmost cells) to save_dir/teff.txt
  void save_teff() {
    Real sums[2] = {0, 0};  // flux, number of cells
    if (boundary_condition::is_physical_boundary(Direction::Z, Side::OUTER,
                                                 mpi_shape)) {
      const int k = grid.k_margin + grid.k_size - 1;
      for (int i = grid.i_margin; i < grid.i_margin + grid.i_size; ++i) {
        for (int j = grid.j_margin; j < grid.j_margin + grid.j_size; ++j) {
          sums[0] += radiation.heating.flux_z(i, j, k);
          sums[1] += 1;
        }
      }
    }
    Real sums_g[2];
    MPI_Allreduce(sums, sums_g, 2, mpi::data_type<Real>(), MPI_SUM, mpi::comm());
    if (mpi::is_root()) {
      const Real teff =
          std::pow(sums_g[0] / sums_g[1] / rt::stefan_boltzmann<Real>, 0.25);
      // a new file for a fresh run
      std::ofstream ofs(fs::path(config.save_dir) / "teff.txt",
                        time.time == Real(0) ? std::ios::trunc : std::ios::app);
      ofs << time.time << " " << teff << "\n";
    }
  }
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
