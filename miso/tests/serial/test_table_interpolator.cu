#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <miso/table_interpolator.hpp>

using namespace miso;

TEST_CASE("Test Table Interpolator in CUDA backend" *
          doctest::test_suite("table_interpolator")) {
  // Table data and interpolator
  constexpr int Nt = 5;
  constexpr float x0 = 0.f;
  constexpr float x1 = static_cast<float>(Nt - 1);
  Array1D<float, backend::Host> table_h(Nt);
  for (int i = 0; i < Nt; ++i) {
    table_h[i] = static_cast<float>(i * i);
  }
  Array1D<float, backend::CUDA> table_d(Nt);
  table_d.copy_from(table_h);
  UniformTableInterpolator1D<float, backend::CUDA> interpolator(table_d.view(),
                                                                x0, x1);

  // Check the interpolator at the table points.
  {
    constexpr int Nx = Nt;
    Array1D<float, backend::Host> out_h(Nx);
    Array1D<float, backend::CUDA> out_d(Nx);
    Array1D<float, backend::Host> in_h(Nx);
    Array1D<float, backend::CUDA> in_d(Nx);
    for (int i = 0; i < Nx; ++i) {
      in_h[i] = static_cast<float>(i);
    }
    in_d.copy_from(in_h);
    interpolator.interpolate(in_d.const_view(), out_d.view());
    out_h.copy_from(out_d);
    for (int i = 0; i < Nx; ++i) {
      REQUIRE(std::abs(out_h[i] - table_h[i]) < 1e-5f);
    }
  }

  // Check the interpolator between the table points.
  {
    constexpr int Nx = Nt - 1;
    Array1D<float, backend::Host> out_h(Nx);
    Array1D<float, backend::CUDA> out_d(Nx);
    Array1D<float, backend::Host> in_h(Nx);
    Array1D<float, backend::CUDA> in_d(Nx);
    for (int i = 0; i < Nx; ++i) {
      in_h[i] = static_cast<float>(i) + 0.5f;
    }
    in_d.copy_from(in_h);
    interpolator.interpolate(in_d.const_view(), out_d.view());
    out_h.copy_from(out_d);
    for (int i = 0; i < Nx; ++i) {
      REQUIRE(std::abs(out_h[i] - 0.5f * (table_h[i] + table_h[i + 1])) < 1e-5f);
    }
  }

  // Check the interpolator outside the table range.
  {
    constexpr int Nx = 2;
    Array1D<float, backend::Host> out_h(Nx);
    Array1D<float, backend::CUDA> out_d(Nx);
    Array1D<float, backend::Host> in_h(Nx);
    Array1D<float, backend::CUDA> in_d(Nx);
    in_h[0] = x0 - 0.5f;
    in_h[1] = x1 + 0.5f;
    in_d.copy_from(in_h);
    interpolator.interpolate(in_d.const_view(), out_d.view());
    out_h.copy_from(out_d);
    REQUIRE(std::abs(out_h[0] - table_h[0]) < 1e-5f);
    REQUIRE(std::abs(out_h[1] - table_h[Nt - 1]) < 1e-5f);
  }
}
