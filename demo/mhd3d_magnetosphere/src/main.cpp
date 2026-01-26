#include <string>

#include <miso/config.hpp>
#include <miso/env.hpp>
#include <miso/eos.hpp>
#include <miso/execution.hpp>
#include <miso/grid.hpp>
#include <miso/mhd.hpp>
#include <miso/mpi_util.hpp>
#include <miso/time.hpp>
#include <miso/types.hpp>
#include <miso/utility.hpp>
#ifdef USE_CUDA
#include <miso/cuda_util.cuh>
#endif

using namespace miso;

#ifdef USE_CUDA
using Backend = backend::CUDA;
#else
using Backend = backend::Host;
#endif

struct InitialCondition {
  Real ro_sw;
  Real pr_sw;
  Real vx_sw;
  Real bz_imf;
  Real pr_earth;

  explicit InitialCondition(Config &config) {
    ro_sw = config.yaml_obj["solar_wind"]["mass_density"].as<Real>();
    pr_sw = config.yaml_obj["solar_wind"]["gas_pressure"].as<Real>();
    vx_sw = config.yaml_obj["solar_wind"]["x_velocity_field"].as<Real>();
    bz_imf = config.yaml_obj["solar_wind"]["z_magnetic_field"].as<Real>();
    pr_earth = config.yaml_obj["magnetosphere"]["gas_pressure"].as<Real>();
  }

  // The signature must not be changed as it is called inside miso::mhd::MHD.
  template <typename EOS>
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid,
             const EOS &eos) const {
    Range3D range{{0, grid.i_total}, {0, grid.j_total}, {0, grid.k_total}};

    for_each(Backend{}, range, MISO_LAMBDA(int i, int j, int k));
    for_each<Backend>(range, MISO_LAMBDA(int i, int j, int k));

    for_each(
        Backend{}, range, MISO_LAMBDA(int i, int j, int k) {
          // distance from the earth center
          Real rr = util::sqrt(grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] +
                               grid.z[k] * grid.z[k]);

          // density
          Real ro_tmp = util::pow3(1 / rr);
          Real ro_sw_thresh = 0.2 * ro_sw;
          if (ro_tmp > ro_sw_thresh) {
            qq.ro(i, j, k) = ro_tmp;
          } else {
            qq.ro(i, j, k) = ro_sw_thresh;
          }

          // dipole magnetic field
          Real rro5 = util::pow5(1 / rr);
          qq.bx(i, j, k) = -rro5 * 3.0 * grid.x[i] * grid.z[k];
          qq.by(i, j, k) = -rro5 * 3.0 * grid.y[j] * grid.z[k];
          qq.bz(i, j, k) =
              +rro5 * (grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] -
                       2.0 * grid.z[k] * grid.z[k]);

          // pressure
          Real pr;
          Real pr_tmp = pr_earth * util::pow2(1 / rr);
          if (pr_tmp > pr_sw) {
            pr = pr_tmp;
          } else {
            pr = pr_sw;
          }
          qq.ei(i, j, k) = pr / (eos.gm - 1.0) / qq.ro(i, j, k);

          qq.vx(i, j, k) = 0.0;
          qq.vy(i, j, k) = 0.0;
          qq.vz(i, j, k) = 0.0;

          qq.ph(i, j, k) = 0.0;
        });
  }
};

struct BoundaryCondition {
  mpi::Shape &mpi_shape;
  Array3D<Real, Backend> f_sphere;
  Array3D<Real, Backend> mask;
  mhd::Fields<Real, Backend> qq_init;

  Real ro_sw;
  Real pr_sw;
  Real vx_sw;
  Real bz_imf;

  Real pr_earth;
  Real rra;
  Real a0;

  Real ro_floor;
  Real pr_floor;

  explicit BoundaryCondition(Config &config, Grid<Real, backend::Host> &grid,
                             mpi::Shape &mpi_shape)
      : mpi_shape(mpi_shape), f_sphere(grid.i_total, grid.j_total, grid.k_total),
        mask(grid.i_total, grid.j_total, grid.k_total),
        qq_init(grid.i_total, grid.j_total, grid.k_total) {
    ro_sw = config.yaml_obj["solar_wind"]["mass_density"].as<Real>();
    pr_sw = config.yaml_obj["solar_wind"]["gas_pressure"].as<Real>();
    vx_sw = config.yaml_obj["solar_wind"]["x_velocity_field"].as<Real>();
    bz_imf = config.yaml_obj["solar_wind"]["z_magnetic_field"].as<Real>();

    pr_earth = config.yaml_obj["magnetosphere"]["gas_pressure"].as<Real>();
    rra = config.yaml_obj["magnetosphere"]["radius"].as<Real>();
    a0 = config.yaml_obj["magnetosphere"]["a0"].as<Real>();

    ro_floor = config.yaml_obj["floor"]["ro_floor"].as<Real>();
    pr_floor = config.yaml_obj["floor"]["pr_floor"].as<Real>();

    // init f_sphere
    // init mask
    // init qq_init
  }

  // The signature must not be changed as it is called by miso integrator.
  template <typename EOS>
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid,
             const EOS &eos) const {
    Backend btag{};

    if (bnd::is_physical_boundary(Direction::X, Side::OUTER, mpi_shape)) {
      bnd::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::X, Side::OUTER);
      bnd::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::X, Side::OUTER);
      bnd::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::X, Side::OUTER);
      bnd::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::X, Side::OUTER);
      bnd::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::X, Side::OUTER);
      bnd::symmetric(btag, qq.by, grid, Sign::Pos, Direction::X, Side::OUTER);
      bnd::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::X, Side::OUTER);
      bnd::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::X, Side::OUTER);
      bnd::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::X, Side::OUTER);
    }

    if (bnd::is_physical_boundary(Direction::Y, Side::INNER, mpi_shape)) {
      bnd::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::Y, Side::INNER);
      bnd::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::Y, Side::INNER);
      bnd::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::Y, Side::INNER);
      bnd::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::Y, Side::INNER);
      bnd::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::Y, Side::INNER);
      bnd::symmetric(btag, qq.by, grid, Sign::Pos, Direction::Y, Side::INNER);
      bnd::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::Y, Side::INNER);
      bnd::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::Y, Side::INNER);
      bnd::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::Y, Side::INNER);
    }

    if (bnd::is_physical_boundary(Direction::Y, Side::OUTER, mpi_shape)) {
      bnd::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bnd::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bnd::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bnd::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bnd::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bnd::symmetric(btag, qq.by, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bnd::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bnd::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bnd::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::Y, Side::OUTER);
    }

    if (bnd::is_physical_boundary(Direction::Z, Side::INNER, mpi_shape)) {
      bnd::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::Z, Side::INNER);
      bnd::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::Z, Side::INNER);
      bnd::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::Z, Side::INNER);
      bnd::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::Z, Side::INNER);
      bnd::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::Z, Side::INNER);
      bnd::symmetric(btag, qq.by, grid, Sign::Pos, Direction::Z, Side::INNER);
      bnd::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::Z, Side::INNER);
      bnd::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::Z, Side::INNER);
      bnd::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::Z, Side::INNER);
    }

    if (bnd::is_physical_boundary(Direction::Z, Side::OUTER, mpi_shape)) {
      bnd::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bnd::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bnd::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bnd::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bnd::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bnd::symmetric(btag, qq.by, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bnd::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bnd::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bnd::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::Z, Side::OUTER);
    }

    // Fix values at the upwind solar wind boundary (inner x-boundary)
    if (bnd::is_physical_boundary(Direction::X, Side::INNER, mpi_shape)) {
      int i0, i1, j0, j1, k0, k1;
      bnd::range_set<Real>(i0, i1, j0, j1, k0, k1, Direction::X, grid);
      Range3D range{{i0, i1}, {j0, j1}, {k0, k1}};
      for_each(
          btag, range, MISO_LAMBDA(int i, int j, int k) {
            int i_ghst, i_trgt;
            bnd::symmetric_index<Real>(i, grid.i_total, grid.i_margin, i_ghst,
                                       i_trgt, Side::INNER);
            qq.ro[grid.idx(i_ghst, j, k)] = ro_sw;
            qq.vx[grid.idx(i_ghst, j, k)] = vx_sw;
            qq.ei[grid.idx(i_ghst, j, k)] =
                pr_sw / (eos.gm - 1.0) / qq.ro[grid.idx(i_ghst, j, k)];
            qq.bz[grid.idx(i_ghst, j, k)] = bz_imf;
          });
    }

    // Fix values near the Earth
    {
      Range3D range{{0, grid.i_total}, {0, grid.j_total}, {0, grid.k_total}};
      auto lerp = [](Real a, Real b, Real f) -> Real {
        return a * f + b * (1.0 - f);
      };
      for_each(
          btag, range, MISO_LAMBDA(int i, int j, int k) {
            Real f = f_sphere(i, j, k);
            qq.ro(i, j, k) = lerp(qq.ro(i, j, k), qq_init.ro(i, j, k), f);
            qq.vx(i, j, k) = lerp(qq.vx(i, j, k), qq_init.vx(i, j, k), f);
            qq.vy(i, j, k) = lerp(qq.vy(i, j, k), qq_init.vy(i, j, k), f);
            qq.vz(i, j, k) = lerp(qq.vz(i, j, k), qq_init.vz(i, j, k), f);
            qq.bx(i, j, k) = lerp(qq.bx(i, j, k), qq_init.bx(i, j, k), f);
            qq.by(i, j, k) = lerp(qq.by(i, j, k), qq_init.by(i, j, k), f);
            qq.bz(i, j, k) = lerp(qq.bz(i, j, k), qq_init.bz(i, j, k), f);
            qq.ei(i, j, k) = lerp(qq.ei(i, j, k), qq_init.ei(i, j, k), f);
            qq.ph(i, j, k) = lerp(qq.ph(i, j, k), qq_init.ph(i, j, k), f);

            qq.ro(i, j, k) = util::max2<Real>(qq.ro(i, j, k), ro_floor);
            qq.ei(i, j, k) = util::max2<Real>(
                qq.ei(i, j, k), pr_floor / (eos.gm - 1.0) / qq.ro(i, j, k));
          });
    }
  }
};

template <typename Real> struct ExternalSources {
  GridView<const Real> grid;
  Real g_grav;

  explicit ExternalSources(Config &config, Grid<Real> &grid_) {
    grid = grid_.const_view();
    g_grav =
        config.yaml_obj["magnetosphere"]["gravitational_acceleration"].as<Real>();
  }

  // External force: x-direction
  // The signature must not be changed as it is called by miso integrator.
  // force is defined in the unit of g/cm^2 s^2 i.e., force per unit volume
  // i.e., acceleration * density
  __host__ __device__ inline Real vx(FieldsView<const Real> qq, int i, int j,
                                     int k) const noexcept {
    return -qq.ro(i, j, k) * g_grav * grid.x[i] /
           util::pow3(util::sqrt(grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] +
                                 grid.z[k] * grid.z[k]));
  }

  // External force: y-direction
  __host__ __device__ inline Real vy(FieldsView<const Real> qq, int i, int j,
                                     int k) const noexcept {
    return -qq.ro(i, j, k) * g_grav * grid.y[j] /
           util::pow3(util::sqrt(grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] +
                                 grid.z[k] * grid.z[k]));
  }

  // External force: z-direction
  __host__ __device__ inline Real vz(FieldsView<const Real> qq, int i, int j,
                                     int k) const noexcept {
    return -qq.ro(i, j, k) * g_grav * grid.z[k] /
           util::pow3(util::sqrt(grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] +
                                 grid.z[k] * grid.z[k]));
  }

  // External heating
  __host__ __device__ inline Real ei(FieldsView<const Real>, int, int,
                                     int) const noexcept {
    return 0.0;
  }
};

struct Model {
  Config &config;
  mpi::Shape mpi_shape;
  Time<Real> time;
  Grid<Real, backend::Host> grid_global;
  Grid<Real, backend::Host> grid;

  mhd::ExecContext<Backend> exec_ctx;
  eos::IdealEOS<Real> eos;
  InitialCondition ic;
  BoundaryCondition bc;
  ExternalSources<Real> src;
  mhd::MHD<Real, eos::IdealEOS<Real>, Backend> mhd;

  Model(Config &config)
      : config(config), mpi_shape(config), time(config), grid_global(config),
        grid(grid_global, mpi_shape), exec_ctx(mpi_shape, grid), eos(config),
        mhd(config, grid, exec_ctx, eos), ic(config), src(config, mhd.grid) {}

  void save_metadata() {
    MPI_Barrier(mpi::comm());
    config.save();
    grid_global.save(config);
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

      if (mpi::is_root()) {
        std::cout << std::fixed << std::setprecision(2)
                  << "time = " << std::setw(6) << time.time
                  << ";  n_step = " << std::setw(8) << time.n_step
                  << ";  n_output = " << std::setw(8) << time.n_output
                  << std::endl;
      }

      time.n_output++;
    }
  }

  /// @brief Main time integration loop
  void run() {
    if (config["base"]["continue"].template as<bool>() &&
        fs::exists(config.time_save_dir + "n_output.txt")) {
      load_state();
    }

    save_metadata();
    mhd.apply_initial_condition(ic, bc);

    MPI_Barrier(mpi::comm());

    save_if_needed();
    while (time.time < time.tend) {
      // basic MHD time integration
      const auto dt = mhd.cfl();
      mhd.update(dt, bc, src);

      // Time is update after all procedures
      time.update(dt);
      save_if_needed();
    }
  }
};

int main(int argc, char *argv[]) {
  using namespace miso;
  std::string config_dir = CONFIG_DIR;

  Env ctx(argc, argv);
  Config config(config_dir + "config.yaml");
  Model model(config);
  model.run();
}
