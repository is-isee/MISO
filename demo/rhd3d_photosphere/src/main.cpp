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

#include <fstream>
#include <string>

#include "common.hpp"

#include "bc.hpp"
#include "ic.hpp"
#include "src.hpp"

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
        strat(input_dir / "stratification_miso.tbl"), ic(config, strat),
        bc(mpi_shape, grid, strat, eos_file), radiation(config, grid),
        gx(gravity(grid, strat)), damping_top(top_layer(config, grid)),
        src(config, radiation.source(), gx, damping_top) {}

  /// @brief RT for the current state, then one MHD step
  void update() {
    radiation.update(mhd.qq, eos, grid, mpi_shape);
    // once after each output (the state of the output)
    if (time.n_output - 1 > n_teff) {
      save_teff();
      n_teff = time.n_output - 1;
    }
    src.set_time(time.time);
    ModelBase::update();
  }

private:
  int n_teff = -1;

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
