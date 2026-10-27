#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <miso/eos_table.hpp>

using namespace miso;

namespace {

constexpr int n_ro = 6, n_en = 5;
constexpr double log_ro_min = -5.0, log_ro_max = 0.0;
constexpr double log_en_min = 20.0, log_en_max = 24.0;

// ln T and ln kappa are linear in (ln rho, ln e), so bilinear interpolation
// in the table is exact.
double log_te_exact(double log_ro, double log_en) {
  return 0.1 * log_ro + 0.5 * log_en - 2.0;
}
double log_op_exact(double log_ro, double log_en) {
  return 0.7 * log_ro - 0.3 * log_en + 1.0;
}

// Write a small table in the r2d2plus-table format.
fs::path write_table(const std::string &kind) {
  const fs::path path =
      fs::temp_directory_path() / ("miso_test_eos_table_" + kind + ".tbl");
  std::vector<double> ro(n_ro), en(n_en), te(n_ro * n_en), op(n_ro * n_en);
  for (int i = 0; i < n_ro; ++i) {
    ro[i] = log_ro_min + (log_ro_max - log_ro_min) * i / (n_ro - 1);
  }
  for (int j = 0; j < n_en; ++j) {
    en[j] = log_en_min + (log_en_max - log_en_min) * j / (n_en - 1);
  }
  for (int i = 0; i < n_ro; ++i) {
    for (int j = 0; j < n_en; ++j) {
      te[i * n_en + j] = std::exp(log_te_exact(ro[i], en[j]));
      op[i * n_en + j] = std::exp(log_op_exact(ro[i], en[j]));
    }
  }

  std::string header = "r2d2plus-table 1\nkind " + kind +
                       "\nlayout row_major little_endian float64\n"
                       "binary_offset 1024\n"
                       "array ro " +
                       std::to_string(n_ro) + "\narray en " +
                       std::to_string(n_en) + "\narray te " +
                       std::to_string(n_ro) + " " + std::to_string(n_en) +
                       "\narray op " + std::to_string(n_ro) + " " +
                       std::to_string(n_en) + "\nmeta {}\n";
  header.resize(1024, '\0');

  std::ofstream ofs(path, std::ios::binary);
  ofs.write(header.data(), header.size());
  for (const auto *v : {&ro, &en, &te, &op}) {
    ofs.write(reinterpret_cast<const char *>(v->data()),
              sizeof(double) * v->size());
  }
  return path;
}

}  // namespace

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

TEST_CASE("TableEOS rejects other kinds of tables" *
          doctest::test_suite("eos_table")) {
  const auto path = write_table("eos_table_sero");
  CHECK_THROWS_AS(eos::TableEOS<double>{path}, std::runtime_error);
  fs::remove(path);
  CHECK_THROWS_AS(eos::TableEOS<double>{path}, std::runtime_error);
}
