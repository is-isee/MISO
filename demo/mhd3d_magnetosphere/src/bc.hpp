#include "common.hpp"

struct BoundaryCondition {
  mpi::Shape &mpi_shape;
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

  explicit BoundaryCondition(Config &config, Grid<Real, Backend> &grid,
                             mpi::Shape &mpi_shape)
      : mpi_shape(mpi_shape), qq_init(grid.i_total, grid.j_total, grid.k_total) {
    ro_sw = config["solar_wind"]["mass_density"].as<Real>();
    pr_sw = config["solar_wind"]["gas_pressure"].as<Real>();
    vx_sw = config["solar_wind"]["x_velocity_field"].as<Real>();
    bz_imf = config["solar_wind"]["z_magnetic_field"].as<Real>();

    pr_earth = config["magnetosphere"]["gas_pressure"].as<Real>();
    rra = config["magnetosphere"]["radius"].as<Real>();
    a0 = config["magnetosphere"]["a0"].as<Real>();

    ro_floor = config["floor"]["ro_floor"].as<Real>();
    pr_floor = config["floor"]["pr_floor"].as<Real>();
  }

  // The signature must not be changed as it is called by miso integrator.
  template <typename EOS>
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid,
             const EOS &eos) const {
    namespace bc = miso::boundary_condition;
    Backend btag{};

    if (bc::is_physical_boundary(Direction::X, Side::OUTER, mpi_shape)) {
      bc::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.by, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::X, Side::OUTER);
      bc::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::X, Side::OUTER);
    }

    if (bc::is_physical_boundary(Direction::Y, Side::INNER, mpi_shape)) {
      bc::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.by, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::Y, Side::INNER);
      bc::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::Y, Side::INNER);
    }

    if (bc::is_physical_boundary(Direction::Y, Side::OUTER, mpi_shape)) {
      bc::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.by, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::Y, Side::OUTER);
      bc::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::Y, Side::OUTER);
    }

    if (bc::is_physical_boundary(Direction::Z, Side::INNER, mpi_shape)) {
      bc::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.by, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::Z, Side::INNER);
      bc::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::Z, Side::INNER);
    }

    if (bc::is_physical_boundary(Direction::Z, Side::OUTER, mpi_shape)) {
      bc::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.by, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::Z, Side::OUTER);
      bc::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::Z, Side::OUTER);
    }

    // Fix values at the upwind solar wind boundary (inner x-boundary)
    if (bc::is_physical_boundary(Direction::X, Side::INNER, mpi_shape)) {
      int i0, i1, j0, j1, k0, k1;
      bc::range_set(i0, i1, j0, j1, k0, k1, Direction::X, grid);
      Range3D range{{i0, i1}, {j0, j1}, {k0, k1}};
      for_each(
          btag, range, MISO_LAMBDA(int i, int j, int k) {
            int i_ghst, i_trgt;
            bc::symmetric_index(i, grid.i_total, grid.i_margin, i_ghst, i_trgt,
                                Side::INNER);
            qq.ro(i_ghst, j, k) = ro_sw;
            qq.vx(i_ghst, j, k) = vx_sw;
            qq.ei(i_ghst, j, k) = pr_sw / (eos.gm - 1.0) / qq.ro(i_ghst, j, k);
            qq.bz(i_ghst, j, k) = bz_imf;
          });
    }

    // Fix values near the Earth
    {
      Range3D range{{0, grid.i_total}, {0, grid.j_total}, {0, grid.k_total}};
      auto lerp = MISO_LAMBDA(Real a, Real b, Real f)->Real {
        return a * f + b * (1.0 - f);
      };
      auto qq_init_v = qq_init.const_view();
      for_each(
          btag, range, MISO_LAMBDA(int i, int j, int k) {
            // Determine blending factor
            Real rr = util::sqrt(grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] +
                                 grid.z[k] * grid.z[k]);
            Real hh = util::max2<Real>(0.0, util::pow2(rr / rra) - 1.0);
            Real f = a0 * hh / (a0 * hh + 1.0);

            // Blend values
            qq.ro(i, j, k) = lerp(qq.ro(i, j, k), qq_init_v.ro(i, j, k), f);
            qq.vx(i, j, k) = lerp(qq.vx(i, j, k), qq_init_v.vx(i, j, k), f);
            qq.vy(i, j, k) = lerp(qq.vy(i, j, k), qq_init_v.vy(i, j, k), f);
            qq.vz(i, j, k) = lerp(qq.vz(i, j, k), qq_init_v.vz(i, j, k), f);
            qq.bx(i, j, k) = lerp(qq.bx(i, j, k), qq_init_v.bx(i, j, k), f);
            qq.by(i, j, k) = lerp(qq.by(i, j, k), qq_init_v.by(i, j, k), f);
            qq.bz(i, j, k) = lerp(qq.bz(i, j, k), qq_init_v.bz(i, j, k), f);
            qq.ei(i, j, k) = lerp(qq.ei(i, j, k), qq_init_v.ei(i, j, k), f);
            qq.ph(i, j, k) = lerp(qq.ph(i, j, k), qq_init_v.ph(i, j, k), f);

            // Apply floors
            qq.ro(i, j, k) = util::max2<Real>(qq.ro(i, j, k), ro_floor);
            qq.ei(i, j, k) = util::max2<Real>(
                qq.ei(i, j, k), pr_floor / (eos.gm - 1.0) / qq.ro(i, j, k));
          });
    }
  }
};
