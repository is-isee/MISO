#pragma once

#include <doctest/doctest.h>

#include <algorithm>
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

// ln T, ln kappa, ln p and ln c_s are linear in (ln rho, ln e), so bilinear
// interpolation in the table is exact.
double log_te_exact(double log_ro, double log_en) {
  return 0.1 * log_ro + 0.5 * log_en - 2.0;
}
double log_op_exact(double log_ro, double log_en) {
  return 0.7 * log_ro - 0.3 * log_en + 1.0;
}
double log_pr_exact(double log_ro, double log_en) {
  return log_ro + log_en - 0.5;
}
double log_cs_exact(double log_ro, double log_en) {
  return -0.05 * log_ro + 0.5 * log_en + 0.2;
}

// Write a small table in the r2d2plus-table format.
// The file name has `tag`, so that the tests running at the same time do not
// write the same file.
fs::path write_table(const std::string &kind, bool uniform = true,
                     const std::string &tag = "cpp") {
  const fs::path path = fs::temp_directory_path() /
                        ("miso_test_eos_table_" + kind + "_" + tag + ".tbl");
  std::vector<double> ro(n_ro), en(n_en), te(n_ro * n_en), op(n_ro * n_en),
      pr(n_ro * n_en), cs(n_ro * n_en);
  for (int i = 0; i < n_ro; ++i) {
    ro[i] = log_ro_min + (log_ro_max - log_ro_min) * i / (n_ro - 1);
  }
  for (int j = 0; j < n_en; ++j) {
    en[j] = log_en_min + (log_en_max - log_en_min) * j / (n_en - 1);
  }
  if (!uniform) {
    ro[2] += 0.1;
  }
  for (int i = 0; i < n_ro; ++i) {
    for (int j = 0; j < n_en; ++j) {
      te[i * n_en + j] = std::exp(log_te_exact(ro[i], en[j]));
      op[i * n_en + j] = std::exp(log_op_exact(ro[i], en[j]));
      pr[i * n_en + j] = std::exp(log_pr_exact(ro[i], en[j]));
      cs[i * n_en + j] = std::exp(log_cs_exact(ro[i], en[j]));
    }
  }

  std::string header =
      "r2d2plus-table 1\nkind " + kind +
      "\nlayout row_major little_endian float64\n"
      "binary_offset 1024\n"
      "array ro " +
      std::to_string(n_ro) + "\narray en " + std::to_string(n_en) +
      "\narray te " + std::to_string(n_ro) + " " + std::to_string(n_en) +
      "\narray op " + std::to_string(n_ro) + " " + std::to_string(n_en) +
      "\narray pr " + std::to_string(n_ro) + " " + std::to_string(n_en) +
      "\narray cs " + std::to_string(n_ro) + " " + std::to_string(n_en) +
      "\nmeta {}\n";
  header.resize(1024, '\0');

  std::ofstream ofs(path, std::ios::binary);
  ofs.write(header.data(), header.size());
  for (const auto *v : {&ro, &en, &te, &op, &pr, &cs}) {
    ofs.write(reinterpret_cast<const char *>(v->data()),
              sizeof(double) * v->size());
  }
  return path;
}

// Pressure and sound speed of the cells (ln rho, ln e) computed by
// TableEOS<double, Backend> on the Backend, copied to the host.
template <typename Backend>
void check_gas_pressure_and_sound_speed(const fs::path &path) {
  const std::vector<double> log_ros = {-4.7, -2.5, -0.1, -10.0, 3.0};
  const std::vector<double> log_ens = {20.3, 22.0, 23.9, 30.0, 10.0};
  const int n = static_cast<int>(log_ros.size());
  Array3D<double> ro_h(n, 1, 1), ei_h(n, 1, 1), pr_h(n, 1, 1), cs_h(n, 1, 1);
  for (int i = 0; i < n; ++i) {
    ro_h(i, 0, 0) = std::exp(log_ros[i]);
    ei_h(i, 0, 0) = std::exp(log_ens[i]);
  }
  Array3D<double, Backend> ro(n, 1, 1), ei(n, 1, 1), pr(n, 1, 1), cs(n, 1, 1);
  ro.copy_from(ro_h);
  ei.copy_from(ei_h);

  eos::TableEOS<double, Backend> eos(path);
  const auto r = ro.const_view();
  const mhd::FieldsView<const double> qq(r, r, r, r, r, r, r, ei.const_view(), r);
  eos.gas_pressure(Backend{}, qq, pr.view());
  eos.sound_speed(Backend{}, qq, cs.view());
  pr_h.copy_from(pr);
  cs_h.copy_from(cs);

  for (int i = 0; i < n; ++i) {
    // The last two cells are outside the table: the value at the nearest
    // edge (two opposite corners).
    const double log_ro = std::min(std::max(log_ros[i], log_ro_min), log_ro_max);
    const double log_en = std::min(std::max(log_ens[i], log_en_min), log_en_max);
    CHECK(std::log(pr_h(i, 0, 0)) ==
          doctest::Approx(log_pr_exact(log_ro, log_en)).epsilon(1e-12));
    CHECK(std::log(cs_h(i, 0, 0)) ==
          doctest::Approx(log_cs_exact(log_ro, log_en)).epsilon(1e-12));
  }
}

}  // namespace
