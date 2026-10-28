#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "test_eos_table_common.hpp"

TEST_CASE("TableEOS interpolates ln T and ln kappa" *
          doctest::test_suite("eos_table")) {
  const auto path = write_table("eos_table_roen");
  eos::TableEOS<double> eos(path);
  REQUIRE(eos.log_te.extent(0) == n_ro);
  REQUIRE(eos.log_te.extent(1) == n_en);

  // Between the table points
  for (double log_ro : {-4.7, -2.5, -0.1}) {
    for (double log_en : {20.3, 22.0, 23.9}) {
      const double ro = std::exp(log_ro), ei = std::exp(log_en);
      CHECK(std::log(eos.temperature(ro, ei)) ==
            doctest::Approx(log_te_exact(log_ro, log_en)).epsilon(1e-12));
      CHECK(std::log(eos.opacity(ro, ei)) ==
            doctest::Approx(log_op_exact(log_ro, log_en)).epsilon(1e-12));
    }
  }

  // Outside the table, the value at the nearest edge is used.
  CHECK(std::log(eos.temperature(std::exp(-10.0), std::exp(30.0))) ==
        doctest::Approx(log_te_exact(log_ro_min, log_en_max)).epsilon(1e-12));

  fs::remove(path);
}

TEST_CASE("TableEOS computes pressure and sound speed on the host" *
          doctest::test_suite("eos_table")) {
  const auto path = write_table("eos_table_roen");
  check_gas_pressure_and_sound_speed<backend::Host>(path);
  fs::remove(path);
}

TEST_CASE("TableEOS rejects other kinds of tables" *
          doctest::test_suite("eos_table")) {
  const auto path = write_table("eos_table_sero");
  CHECK_THROWS_AS(eos::TableEOS<double>{path}, std::runtime_error);
  fs::remove(path);
  CHECK_THROWS_AS(eos::TableEOS<double>{path}, std::runtime_error);
}

TEST_CASE("TableEOS rejects non-uniform axes" *
          doctest::test_suite("eos_table")) {
  const auto path = write_table("eos_table_roen", false);
  CHECK_THROWS_AS(eos::TableEOS<double>{path}, std::runtime_error);
  fs::remove(path);
}

TEST_CASE("TableEOS reads the table in input_data" *
          doctest::test_suite("eos_table")) {
  const fs::path path = fs::path(INPUT_DATA_DIR) / "eos_table_roen_blend.tbl";
  eos::TableEOS<double> eos(path);
  REQUIRE(eos.log_te.extent(0) == 256);
  REQUIRE(eos.log_te.extent(1) == 256);

  // A table point in the photosphere (values read from the file with numpy)
  const double ro = std::exp(-15.377656523015137);
  const double ei = std::exp(27.981135057266535);
  CHECK(eos.temperature(ro, ei) ==
        doctest::Approx(3296.3360304098305).epsilon(1e-10));
  CHECK(eos.opacity(ro, ei) ==
        doctest::Approx(0.04658069461584091).epsilon(1e-10));

  Array3D<double> ro_a(1, 1, 1), ei_a(1, 1, 1), pr(1, 1, 1), cs(1, 1, 1);
  ro_a(0, 0, 0) = ro;
  ei_a(0, 0, 0) = ei;
  const auto r = ro_a.const_view();
  const mhd::FieldsView<const double> qq(r, r, r, r, r, r, r, ei_a.const_view(),
                                         r);
  eos.gas_pressure(backend::Host{}, qq, pr.view());
  eos.sound_speed(backend::Host{}, qq, cs.view());
  CHECK(pr(0, 0, 0) == doctest::Approx(39138.30864711022).epsilon(1e-10));
  CHECK(cs(0, 0, 0) == doctest::Approx(463724.90858549863).epsilon(1e-10));

  // The opacity is inf here (rho = 1.1e-11 g/cm^3, T = 2.1e6 K) and kept so.
  CHECK(std::isinf(eos.opacity(1.0944997993195436e-11, 460706556163877.0)));
}
