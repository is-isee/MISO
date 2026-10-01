// Petschek-type magnetic reconnection with a localized resistivity
// (Zenitani 2015, Phys. Plasmas 22, 032114).
//
// The x-z plane is solved (x: outflow, z: inflow, y: out of plane) in one
// quadrant [0, x_max] x [0, z_max] with the X-point at the origin. Variables
// are in the Gauss units of MISO: B = sqrt(4 pi) B_paper.

#include <miso/core.hpp>
#include <miso/mhd.hpp>
#include <miso/mhd_resistivity.hpp>

using namespace miso;

using Real = float;

#ifdef USE_CUDA
using Backend = backend::CUDA;
#else
using Backend = backend::Host;
#endif

/// @brief Parameters of the problem (in the units of the paper)
struct Parameters {
  /// @brief plasma beta in the upstream region
  Real beta_up;
  /// @brief background resistivity
  Real eta0;
  /// @brief resistivity at the X-point
  Real eta1;
  /// @brief amplitude of the perturbation of the vector potential A_y
  Real a_pert;

  explicit Parameters(Config &config)
      : beta_up(config["reconnection"]["beta_up"].as<Real>()),
        eta0(config["reconnection"]["eta0"].as<Real>()),
        eta1(config["reconnection"]["eta1"].as<Real>()),
        a_pert(config["reconnection"]["a_pert"].as<Real>()) {}

  /// @brief Localized resistivity eta(x, z)
  Real eta(Real x, Real z) const {
    const Real ch = std::cosh(std::sqrt(x * x + z * z));
    return eta0 + (eta1 - eta0) / (ch * ch);
  }
};

struct InitialCondition {
  eos::IdealEOS<Real> &eos;
  const Parameters &par;

  // The signature must not be changed as it is called inside miso::mhd::MHD.
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {
    const Real b_unit = std::sqrt(Real(4) * pi<Real>);
    const Real gm = eos.gm;
    for (int k = 0; k < grid.k_total; ++k) {
      for (int j = 0; j < grid.j_total; ++j) {
        for (int i = 0; i < grid.i_total; ++i) {
          const Real x = grid.x[i], z = grid.z[k];

          // Harris sheet
          const Real ch = std::cosh(z);
          const Real sech2 = Real(1) / (ch * ch);
          const Real ro = Real(1) + sech2 / par.beta_up;
          const Real pr = Real(0.5) * (par.beta_up + sech2);

          // Perturbation: B = curl(dA_y e_y), dA_y = -a exp[-(x^2 + z^2)/4]
          const Real ex = std::exp(-(x * x + z * z) / Real(4));
          const Real bx = std::tanh(z) - Real(0.5) * par.a_pert * z * ex;
          const Real bz = Real(0.5) * par.a_pert * x * ex;

          qq.ro(i, j, k) = ro;
          qq.vx(i, j, k) = Real(0);
          qq.vy(i, j, k) = Real(0);
          qq.vz(i, j, k) = Real(0);
          qq.bx(i, j, k) = b_unit * bx;
          qq.by(i, j, k) = Real(0);
          qq.bz(i, j, k) = b_unit * bz;
          qq.ei(i, j, k) = pr / (gm - Real(1)) / ro;
          qq.ph(i, j, k) = Real(0);
        }
      }
    }
  }
};

/// @brief Boundaries of the quadrant
/// @details
/// - x = 0, x = x_max, and z = 0: mirror symmetry. The velocity normal to the
///   boundary, the tangential magnetic field (a pseudovector), and the GLM
///   potential (proportional to div B) are odd.
/// - z = z_max: reflecting (perfectly conducting) wall. The normal components
///   of the velocity and the magnetic field are odd.
struct BoundaryCondition {
  mpi::Shape &mpi_shape;

  // The signature must not be changed as it is called by miso integrator.
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {
    constexpr Sign P = Sign::Pos, N = Sign::Neg;
    //                                           vx vy vz bx by bz ph
    reflect(qq, grid, Direction::X, Side::INNER, N, P, P, P, N, N, N);
    reflect(qq, grid, Direction::X, Side::OUTER, N, P, P, P, N, N, N);
    reflect(qq, grid, Direction::Z, Side::INNER, P, P, N, N, N, P, N);
    reflect(qq, grid, Direction::Z, Side::OUTER, P, P, N, P, P, N, P);
  }

  /// @brief Symmetric boundary with the given sign of each variable
  /// (density and internal energy are even)
  void reflect(mhd::FieldsView<Real> qq, GridView<const Real> grid, Direction dir,
               Side side, Sign vx, Sign vy, Sign vz, Sign bx, Sign by, Sign bz,
               Sign ph) const {
    namespace bc = miso::boundary_condition;
    if (!bc::is_physical_boundary(dir, side, mpi_shape)) {
      return;
    }
    Backend btag{};
    bc::symmetric(btag, qq.ro, grid, Sign::Pos, dir, side);
    bc::symmetric(btag, qq.vx, grid, vx, dir, side);
    bc::symmetric(btag, qq.vy, grid, vy, dir, side);
    bc::symmetric(btag, qq.vz, grid, vz, dir, side);
    bc::symmetric(btag, qq.bx, grid, bx, dir, side);
    bc::symmetric(btag, qq.by, grid, by, dir, side);
    bc::symmetric(btag, qq.bz, grid, bz, dir, side);
    bc::symmetric(btag, qq.ei, grid, Sign::Pos, dir, side);
    bc::symmetric(btag, qq.ph, grid, ph, dir, side);
  }
};

/// @brief Resistivity at cell centers (including ghost cells)
inline Array3D<Real, Backend> make_eta(const Grid<Real, backend::Host> &grid,
                                       const Parameters &par) {
  const auto g = grid.const_view();
  Array3D<Real, backend::Host> eta_h(g.i_total, g.j_total, g.k_total);
  for (int k = 0; k < g.k_total; ++k) {
    for (int j = 0; j < g.j_total; ++j) {
      for (int i = 0; i < g.i_total; ++i) {
        eta_h(i, j, k) = par.eta(g.x[i], g.z[k]);
      }
    }
  }
  Array3D<Real, Backend> eta(g.i_total, g.j_total, g.k_total);
  eta.copy_from(eta_h);
  return eta;
}

struct Model : public mhd::ModelBase<Model, Real, Backend> {
  eos::IdealEOS<Real> eos;
  Parameters par;
  InitialCondition ic;
  BoundaryCondition bc;
  Array3D<Real, Backend> eta;
  mhd::ResistiveSource<Real, Backend> src;

  Model(Config &config)
      : ModelBase(config), eos(config), par(config), ic{eos, par}, bc{mpi_shape},
        eta(make_eta(grid, par)), src(config, mhd.grid, eta) {}
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
