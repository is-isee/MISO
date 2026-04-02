#pragma once
#include <miso/boundary_condition.hpp>
#include <miso/mhd_model_base.hpp>

using namespace miso;
#ifdef USE_CUDA
using Backend = backend::CUDA;
#else
using Backend = backend::Host;
#endif

template <typename Real> struct BoundaryCondition {
  mpi::Shape &mpi_shape;

  BoundaryCondition(mpi::Shape &mpi_shape) : mpi_shape(mpi_shape) {}

  // The signature must not be changed as it is called by miso integrator.
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {
    namespace bc = miso::boundary_condition;
    Backend btag{};

    if (bc::is_physical_boundary(Direction::X, Side::INNER, mpi_shape)) {
      bc::symmetric(btag, qq.ro, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.vx, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.vy, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.vz, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.bx, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.by, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.bz, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.ei, grid, Sign::Pos, Direction::X, Side::INNER);
      bc::symmetric(btag, qq.ph, grid, Sign::Pos, Direction::X, Side::INNER);
    }

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
  }
};
