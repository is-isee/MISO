#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <miso/table_interpolator.hpp>

using namespace miso;

TEST_CASE("Test UniformTableInterpolator1D in Host backend" *
          doctest::test_suite("table_interpolator")) {
  // Table data and interpolator
  constexpr int Nt = 5;
  constexpr float x_min = 0.f;
  constexpr float x_max = static_cast<float>(Nt - 1);
  Array1D<float, backend::Host> table_h(Nt);
  for (int i = 0; i < Nt; ++i) {
    table_h[i] = static_cast<float>(i * i);
  }
  UniformTableInterpolator1D<float, backend::Host> interpolator(table_h.view(),
                                                                x_min, x_max);

  // Check the interpolator at the table points.
  {
    constexpr int Nx = Nt;
    Array1D<float, backend::Host> out_h(Nx);
    Array1D<float, backend::Host> in_h(Nx);
    for (int i = 0; i < Nx; ++i) {
      in_h[i] = static_cast<float>(i);
    }
    interpolator.interpolate(in_h.const_view(), out_h.view());
    for (int i = 0; i < Nx; ++i) {
      REQUIRE(std::abs(out_h[i] - table_h[i]) < 1e-5f);
    }
  }

  // Check the interpolator between the table points.
  {
    constexpr int Nx = Nt - 1;
    Array1D<float, backend::Host> out_h(Nx);
    Array1D<float, backend::Host> in_h(Nx);
    for (int i = 0; i < Nx; ++i) {
      in_h[i] = static_cast<float>(i) + 0.5f;
    }
    interpolator.interpolate(in_h.const_view(), out_h.view());
    for (int i = 0; i < Nx; ++i) {
      REQUIRE(std::abs(out_h[i] - 0.5f * (table_h[i] + table_h[i + 1])) < 1e-5f);
    }
  }

  // Check the interpolator outside the table range.
  {
    constexpr int Nx = 2;
    Array1D<float, backend::Host> out_h(Nx);
    Array1D<float, backend::Host> in_h(Nx);
    in_h[0] = x_min - 0.5f;
    in_h[1] = x_max + 0.5f;
    interpolator.interpolate(in_h.const_view(), out_h.view());
    REQUIRE(std::abs(out_h[0] - table_h[0]) < 1e-5f);
    REQUIRE(std::abs(out_h[1] - table_h[Nt - 1]) < 1e-5f);
  }
}

TEST_CASE("Test UniformTableInterpolator2D in Host backend" *
          doctest::test_suite("table_interpolator")) {
  // Table data and interpolator
  constexpr int Nt0 = 2;
  constexpr int Nt1 = 3;
  constexpr float x0_min = 0.f;
  constexpr float x0_max = static_cast<float>(Nt0 - 1);
  constexpr float x1_min = 0.f;
  constexpr float x1_max = static_cast<float>(Nt1 - 1);
  Array2D<float, backend::Host> table_h(Nt0, Nt1);
  for (int i = 0; i < Nt0; ++i) {
    for (int j = 0; j < Nt1; ++j) {
      table_h(i, j) = static_cast<float>(100 * i * i + j * j);
    }
  }
  UniformTableInterpolator2D<float, backend::Host> interpolator(
      table_h.view(), x0_min, x0_max, x1_min, x1_max);

  // Check the interpolator at the table points.
  {
    constexpr int Nx0 = Nt0;
    constexpr int Nx1 = Nt1;
    Array3D<float, backend::Host> out_h(Nx0, Nx1, 1);
    Array3D<float, backend::Host> in0_h(Nx0, Nx1, 1);
    Array3D<float, backend::Host> in1_h(Nx0, Nx1, 1);
    for (int i = 0; i < Nx0; ++i) {
      for (int j = 0; j < Nx1; ++j) {
        in0_h(i, j, 0) = static_cast<float>(i);
        in1_h(i, j, 0) = static_cast<float>(j);
      }
    }
    interpolator.interpolate(in0_h.const_view(), in1_h.const_view(),
                             out_h.view());
    for (int i = 0; i < Nx0 * Nx1; ++i) {
      REQUIRE(std::abs(out_h[i] - table_h[i]) < 1e-5f);
    }
  }

  // Check the interpolator between the table points.
  {
    constexpr int Nx0 = Nt0 - 1;
    constexpr int Nx1 = Nt1 - 1;
    Array3D<float, backend::Host> out_h(Nx0, Nx1, 1);
    Array3D<float, backend::Host> in0_h(Nx0, Nx1, 1);
    Array3D<float, backend::Host> in1_h(Nx0, Nx1, 1);
    for (int i = 0; i < Nx0; ++i) {
      for (int j = 0; j < Nx1; ++j) {
        in0_h(i, j, 0) = static_cast<float>(i) + 0.5f;
        in1_h(i, j, 0) = static_cast<float>(j) + 0.5f;
      }
    }
    interpolator.interpolate(in0_h.const_view(), in1_h.const_view(),
                             out_h.view());
    for (int i = 0; i < Nx0; ++i) {
      for (int j = 0; j < Nx1; ++j) {
        const float expected =
            0.25f * (table_h(i, j) + table_h(i + 1, j) + table_h(i, j + 1) +
                     table_h(i + 1, j + 1));
        REQUIRE(std::abs(out_h(i, j, 0) - expected) < 1e-5f);
      }
    }
  }

  // Check the interpolator outside the table range.
  {
    constexpr int Nx0 = 2;
    constexpr int Nx1 = 2;
    Array3D<float, backend::Host> out_h(Nx0, Nx1, 1);
    Array3D<float, backend::Host> in0_h(Nx0, Nx1, 1);
    Array3D<float, backend::Host> in1_h(Nx0, Nx1, 1);
    in0_h(0, 0, 0) = x0_min - 0.5f;
    in0_h(0, 1, 0) = x0_min - 0.5f;
    in0_h(1, 0, 0) = x0_max + 0.5f;
    in0_h(1, 1, 0) = x0_max + 0.5f;
    in1_h(0, 0, 0) = x1_min - 0.5f;
    in1_h(0, 1, 0) = x1_max + 0.5f;
    in1_h(1, 0, 0) = x1_min - 0.5f;
    in1_h(1, 1, 0) = x1_max + 0.5f;
    interpolator.interpolate(in0_h.const_view(), in1_h.const_view(),
                             out_h.view());
    REQUIRE(std::abs(out_h(0, 0, 0) - table_h(0, 0)) < 1e-5f);
    REQUIRE(std::abs(out_h(0, 1, 0) - table_h(0, Nt1 - 1)) < 1e-5f);
    REQUIRE(std::abs(out_h(1, 0, 0) - table_h(Nt0 - 1, 0)) < 1e-5f);
    REQUIRE(std::abs(out_h(1, 1, 0) - table_h(Nt0 - 1, Nt1 - 1)) < 1e-5f);
  }
}
