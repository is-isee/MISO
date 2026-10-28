#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <miso/eos_table.hpp>
#include <miso/stratification.hpp>

using namespace miso;

namespace {

// Non-uniform z. On the table points, ln ro, ln en, ln pr, ln te, se and gx
// are a linear function of z plus a bump, so that the interpolation from a
// wrong interval gives a wrong value.
const std::vector<double> zs = {-2.0, -1.0, 0.5, 3.0};
const std::vector<double> bumps = {0.0, 1.0, 0.0, 2.0};
// The bump interpolated linearly in z (by hand)
const std::vector<std::pair<double, double>> bump_at = {
    {-2.0, 0.0}, {-1.5, 0.5}, {-1.0, 1.0}, {-0.25, 0.5},
    {0.5, 0.0},  {1.75, 1.0}, {3.0, 2.0}};
double log_ro_exact(double z) { return -1.5 * z - 15.0; }
double log_en_exact(double z) { return -0.2 * z + 28.0; }
double log_pr_exact(double z) { return -1.7 * z + 11.0; }
double log_te_exact(double z) { return -0.1 * z + 8.7; }
double se_exact(double z) { return 2.0e7 * z + 1.6e9; }
double gx_exact(double z) { return -3.0 * z + 2.74e4; }

// Write a small table of kind `kind` in the r2d2plus-table format.
fs::path write_table(const std::string &kind, std::vector<double> z = zs) {
  const fs::path path =
      fs::temp_directory_path() / ("miso_test_stratification_" + kind + ".tbl");
  const int n = static_cast<int>(z.size());
  std::vector<double> ro(n), en(n), pr(n), te(n), se(n), gx(n);
  for (int i = 0; i < n; ++i) {
    const double b = bumps[i];
    ro[i] = std::exp(log_ro_exact(z[i]) + b);
    en[i] = std::exp(log_en_exact(z[i]) + 0.1 * b);
    pr[i] = std::exp(log_pr_exact(z[i]) + b);
    te[i] = std::exp(log_te_exact(z[i]) + 0.1 * b);
    se[i] = se_exact(z[i]) + 1.0e7 * b;
    gx[i] = gx_exact(z[i]) + 100.0 * b;
  }
  std::string header = "r2d2plus-table 1\nkind " + kind +
                       "\nlayout row_major little_endian float64\n"
                       "binary_offset 1024\n";
  for (const char *name : {"z", "ro", "en", "pr", "te", "se", "gx"}) {
    header += std::string("array ") + name + " " + std::to_string(n) + "\n";
  }
  header += "meta {}\nend\n";
  header.resize(1024, '\n');

  std::ofstream ofs(path, std::ios::binary);
  ofs.write(header.data(), header.size());
  for (const auto *v : {&z, &ro, &en, &pr, &te, &se, &gx}) {
    ofs.write(reinterpret_cast<const char *>(v->data()),
              sizeof(double) * v->size());
  }
  return path;
}

}  // namespace

TEST_CASE("Stratification interpolates linearly in z" *
          doctest::test_suite("stratification")) {
  const auto path = write_table("stratification_z");
  const Stratification<double> strat(path);
  REQUIRE(strat.z.size() == zs.size());

  // On the table points (including both ends) and between them
  for (const auto &[z, b] : bump_at) {
    const auto s = strat.at(z);
    CHECK(std::log(s.ro) == doctest::Approx(log_ro_exact(z) + b).epsilon(1e-12));
    CHECK(std::log(s.ei) ==
          doctest::Approx(log_en_exact(z) + 0.1 * b).epsilon(1e-12));
    CHECK(std::log(s.pr) == doctest::Approx(log_pr_exact(z) + b).epsilon(1e-12));
    CHECK(std::log(s.te) ==
          doctest::Approx(log_te_exact(z) + 0.1 * b).epsilon(1e-12));
    CHECK(s.se == doctest::Approx(se_exact(z) + 1.0e7 * b).epsilon(1e-12));
    CHECK(s.gx == doctest::Approx(gx_exact(z) + 100.0 * b).epsilon(1e-12));
  }

  // Outside the table
  CHECK_THROWS_AS(strat.at(-2.1), std::out_of_range);
  CHECK_THROWS_AS(strat.at(3.1), std::out_of_range);
  fs::remove(path);
}

TEST_CASE("Stratification rejects other tables" *
          doctest::test_suite("stratification")) {
  const auto path_kind = write_table("eos_table_roen");
  CHECK_THROWS_AS(Stratification<double>{path_kind}, std::runtime_error);
  fs::remove(path_kind);

  const auto path_order = write_table("stratification_z", {-2.0, 0.5, -1.0, 3.0});
  CHECK_THROWS_AS(Stratification<double>{path_order}, std::runtime_error);
  fs::remove(path_order);
}

TEST_CASE("Stratification reads the table in input_data" *
          doctest::test_suite("stratification")) {
  const Stratification<double> strat(fs::path(INPUT_DATA_DIR) /
                                     "stratification_miso.tbl");
  REQUIRE(strat.z.size() == 1782);

  // At z = 0 (r = Rstar), a table point (values read from the file with numpy)
  const auto s = strat.at(0.0);
  CHECK(s.ro == doctest::Approx(2.027233242128408e-07).epsilon(1e-12));
  CHECK(s.ei == doctest::Approx(2159500990771.7295).epsilon(1e-12));
  CHECK(s.pr == doctest::Approx(78221.79390072174).epsilon(1e-12));
  CHECK(s.te == doctest::Approx(5850.176561215182).epsilon(1e-12));
  CHECK(s.gx == doctest::Approx(27398.31713369527).epsilon(1e-12));

  // The pressure of the EoS table at (ro, ei) of the stratification agrees
  // with pr of the stratification within 0.3% in the photosphere
  // (-5 Mm <= z <= 700 km; see input_data/README.md).
  const eos::TableEOS<double> eos(fs::path(INPUT_DATA_DIR) /
                                  "eos_table_roen_blend.tbl");
  const int n = 121;
  Array3D<double> ro(n, 1, 1), ei(n, 1, 1), pr(n, 1, 1);
  for (int i = 0; i < n; ++i) {
    const auto si = strat.at(-5.0e8 + 5.7e8 * i / (n - 1));
    ro(i, 0, 0) = si.ro;
    ei(i, 0, 0) = si.ei;
  }
  const auto r = ro.const_view();
  const mhd::FieldsView<const double> qq(r, r, r, r, r, r, r, ei.const_view(), r);
  eos.gas_pressure(backend::Host{}, qq, pr.view());
  for (int i = 0; i < n; ++i) {
    const auto si = strat.at(-5.0e8 + 5.7e8 * i / (n - 1));
    CHECK(std::abs(pr(i, 0, 0) / si.pr - 1.0) < 3e-3);
  }
}
