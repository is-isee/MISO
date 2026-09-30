// Petschek-type magnetic reconnection with a localized resistivity
// (Zenitani 2015, Phys. Plasmas 22, 032114).
//
// The paper solves the x-z plane (x: outflow, z: inflow, y: out of plane) in
// one quadrant [0, x_max] x [0, z_max] with the X-point at the origin. Here the
// paper's axes (x, y, z) are mapped to MISO's axes by a cyclic permutation,
// selected by the axis of size 1 (the paper's y):
//
//   size-1 axis  paper (x, y, z) -> MISO   config
//   y            (x, y, z)                 config_zx.yaml
//   z            (y, z, x)                 config_xy.yaml
//   x            (z, x, y)                 config_yz.yaml
//
// A cyclic permutation is a rotation, so the three runs solve the same
// problem. Variables are in the Gauss units of MISO: B = sqrt(4 pi) B_paper.

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

/// @brief MISO axis (0: x, 1: y, 2: z) of each axis of the paper
struct Axes {
  /// @brief outflow direction (paper's x)
  int x;
  /// @brief out-of-plane direction (paper's y)
  int y;
  /// @brief inflow direction (paper's z)
  int z;

  explicit Axes(const Grid<Real, backend::Host> &grid) {
    const int strides[3] = {grid.is, grid.js, grid.ks};
    if (strides[0] + strides[1] + strides[2] != 2) {
      throw std::runtime_error("Exactly one of i/j/k_size must be 1.");
    }
    y = strides[0] == 0 ? 0 : (strides[1] == 0 ? 1 : 2);
    x = (y + 2) % 3;
    z = (y + 1) % 3;
  }
};

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

/// @brief Coordinates of cell (i, j, k) along the MISO axes
inline void position(GridView<const Real> grid, int i, int j, int k,
                     Real (&pos)[3]) {
  pos[0] = grid.x[i];
  pos[1] = grid.y[j];
  pos[2] = grid.z[k];
}

struct InitialCondition {
  eos::IdealEOS<Real> &eos;
  const Axes &axes;
  const Parameters &par;

  // The signature must not be changed as it is called inside miso::mhd::MHD.
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {
    const Real b_unit = std::sqrt(Real(4) * pi<Real>);
    const Real gm = eos.gm;
    for (int k = 0; k < grid.k_total; ++k) {
      for (int j = 0; j < grid.j_total; ++j) {
        for (int i = 0; i < grid.i_total; ++i) {
          Real pos[3];
          position(grid, i, j, k, pos);
          const Real x = pos[axes.x], z = pos[axes.z];

          // Harris sheet
          const Real ch = std::cosh(z);
          const Real sech2 = Real(1) / (ch * ch);
          const Real ro = Real(1) + sech2 / par.beta_up;
          const Real pr = Real(0.5) * (par.beta_up + sech2);

          // Perturbation: B = curl(dA_y e_y), dA_y = -a exp[-(x^2 + z^2)/4]
          const Real ex = std::exp(-(x * x + z * z) / Real(4));
          Real bb[3];
          bb[axes.x] = std::tanh(z) - Real(0.5) * par.a_pert * z * ex;
          bb[axes.y] = Real(0);
          bb[axes.z] = Real(0.5) * par.a_pert * x * ex;

          qq.ro(i, j, k) = ro;
          qq.vx(i, j, k) = Real(0);
          qq.vy(i, j, k) = Real(0);
          qq.vz(i, j, k) = Real(0);
          qq.bx(i, j, k) = b_unit * bb[0];
          qq.by(i, j, k) = b_unit * bb[1];
          qq.bz(i, j, k) = b_unit * bb[2];
          qq.ei(i, j, k) = pr / (gm - Real(1)) / ro;
          qq.ph(i, j, k) = Real(0);
        }
      }
    }
  }
};

/// @brief Boundaries of the quadrant
/// @details
/// - Paper's x = 0, x = x_max, and z = 0: mirror symmetry. The velocity normal
///   to the boundary, the tangential magnetic field (a pseudovector), and the
///   GLM potential (proportional to div B) are odd.
/// - Paper's z = z_max: reflecting (perfectly conducting) wall. The normal
///   components of the velocity and the magnetic field are odd.
struct BoundaryCondition {
  mpi::Shape &mpi_shape;
  const Axes &axes;

  // The signature must not be changed as it is called by miso integrator.
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {
    reflect(qq, grid, axes.x, Side::INNER, true);
    reflect(qq, grid, axes.x, Side::OUTER, true);
    reflect(qq, grid, axes.z, Side::INNER, true);
    reflect(qq, grid, axes.z, Side::OUTER, false);
  }

  /// @brief Mirror symmetry (mirror = true) or reflecting wall (false) at the
  /// boundary normal to the MISO axis `normal`
  void reflect(mhd::FieldsView<Real> qq, GridView<const Real> grid, int normal,
               Side side, bool mirror) const {
    namespace bc = miso::boundary_condition;
    const Direction dir = static_cast<Direction>(normal);
    if (!bc::is_physical_boundary(dir, side, mpi_shape)) {
      return;
    }
    Backend btag{};
    const auto sign = [](bool odd) { return odd ? Sign::Neg : Sign::Pos; };
    Array3DView<Real> vv[3] = {qq.vx, qq.vy, qq.vz};
    Array3DView<Real> bb[3] = {qq.bx, qq.by, qq.bz};

    bc::symmetric(btag, qq.ro, grid, Sign::Pos, dir, side);
    bc::symmetric(btag, qq.ei, grid, Sign::Pos, dir, side);
    bc::symmetric(btag, qq.ph, grid, sign(mirror), dir, side);
    for (int a = 0; a < 3; ++a) {
      const bool is_normal = (a == normal);
      bc::symmetric(btag, vv[a], grid, sign(is_normal), dir, side);
      bc::symmetric(btag, bb[a], grid, sign(mirror != is_normal), dir, side);
    }
  }
};

/// @brief Resistivity at cell centers (including ghost cells)
inline Array3D<Real, Backend> make_eta(const Grid<Real, backend::Host> &grid,
                                       const Axes &axes, const Parameters &par) {
  const auto g = grid.const_view();
  Array3D<Real, backend::Host> eta_h(g.i_total, g.j_total, g.k_total);
  for (int k = 0; k < g.k_total; ++k) {
    for (int j = 0; j < g.j_total; ++j) {
      for (int i = 0; i < g.i_total; ++i) {
        Real pos[3];
        position(g, i, j, k, pos);
        eta_h(i, j, k) = par.eta(pos[axes.x], pos[axes.z]);
      }
    }
  }
  Array3D<Real, Backend> eta(g.i_total, g.j_total, g.k_total);
  eta.copy_from(eta_h);
  return eta;
}

struct Model : public mhd::ModelBase<Model, Real, Backend> {
  eos::IdealEOS<Real> eos;
  Axes axes;
  Parameters par;
  InitialCondition ic;
  BoundaryCondition bc;
  Array3D<Real, Backend> eta;
  mhd::ResistiveSource<Real, Backend> src;

  Model(Config &config)
      : ModelBase(config), eos(config), axes(grid), par(config),
        ic{eos, axes, par}, bc{mpi_shape, axes}, eta(make_eta(grid, axes, par)),
        src(config, mhd.grid, eta) {}
};

int main(int argc, char **argv) {
  // Initialize MPI and CUDA environments
  Env env(argc, argv);

  // Read configuration file
  auto config_path = parse_config_filepath(argc, argv);
  Config config(config_path.value_or("./config/config_zx.yaml"));

  // Run simulation
  Model model(config);
  model.run();
}
