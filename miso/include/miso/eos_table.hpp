#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "array2d.hpp"
#include "table_interpolator.hpp"
#include "utility.hpp"

namespace miso {
namespace eos {

/// @brief Arrays read from a table file in the r2d2plus-table format.
/// @details The file starts with a text header and the arrays follow at
/// `binary_offset` as little-endian float64 in row-major order:
///
///     r2d2plus-table 1
///     kind eos_table_roen
///     layout row_major little_endian float64
///     binary_offset 4096
///     array ro 256
///     array te 256 256
///     ...
///
/// The arrays are stored in the order of the `array` lines.
/// See `input_data/README.md` for the tables in this repository.
struct TableFile {
  /// @brief Value of the `kind` line
  std::string kind;
  /// @brief Shape of each array
  std::map<std::string, std::vector<int>> shapes;
  /// @brief Values of each array
  std::map<std::string, std::vector<double>> arrays;

  explicit TableFile(const fs::path &path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to open table file: " + path.string());
    }

    std::string line;
    if (!std::getline(ifs, line) || line.rfind("r2d2plus-table", 0) != 0) {
      throw std::runtime_error("Not an r2d2plus-table file: " + path.string());
    }

    std::int64_t binary_offset = -1;
    std::vector<std::string> names;
    while (std::getline(ifs, line)) {
      std::istringstream iss(line);
      std::string key;
      iss >> key;
      if (key == "kind") {
        iss >> kind;
      } else if (key == "layout") {
        std::string order, endian, type;
        iss >> order >> endian >> type;
        if (order != "row_major" || endian != "little_endian" ||
            type != "float64") {
          throw std::runtime_error("Unsupported table layout: " + line);
        }
      } else if (key == "binary_offset") {
        iss >> binary_offset;
      } else if (key == "array") {
        std::string name;
        iss >> name;
        std::vector<int> shape;
        int n;
        while (iss >> n) {
          shape.push_back(n);
        }
        names.push_back(name);
        shapes[name] = shape;
      } else if (key == "meta") {
        break;  // the last line of the header
      }
    }
    if (binary_offset < 0 || names.empty()) {
      throw std::runtime_error("Invalid table header: " + path.string());
    }

    const std::uint16_t one = 1;
    if (*reinterpret_cast<const std::uint8_t *>(&one) != 1) {
      throw std::runtime_error("Reading tables on big-endian hosts is not "
                               "supported");
    }

    ifs.clear();
    ifs.seekg(binary_offset);
    for (const auto &name : names) {
      std::size_t size = 1;
      for (int n : shapes[name]) {
        size *= static_cast<std::size_t>(n);
      }
      std::vector<double> values(size);
      ifs.read(reinterpret_cast<char *>(values.data()),
               static_cast<std::streamsize>(sizeof(double) * size));
      if (!ifs) {
        throw std::runtime_error("Failed to read array '" + name +
                                 "' from: " + path.string());
      }
      arrays[name] = std::move(values);
    }
  }

  /// @brief Get an array, checking its shape.
  const std::vector<double> &get(const std::string &name,
                                 const std::vector<int> &shape) const {
    const auto it = shapes.find(name);
    if (it == shapes.end() || it->second != shape) {
      throw std::runtime_error("Table has no array '" + name +
                               "' of the expected shape");
    }
    return arrays.at(name);
  }
};

/// @brief Temperature and opacity from a table in (ln rho, ln e).
/// @details Reads a table of kind `eos_table_roen` (for example,
/// `input_data/eos_table_roen_blend.tbl`). The axes `ro` and `en` are ln(rho
/// [g/cm^3]) and ln(e [erg/g]) on uniform grids. The logarithms of temperature
/// `te` [K] and Rosseland mean opacity `op` [cm^2/g] are interpolated
/// bilinearly. Outside the table range, the value at the nearest edge is used.
///
/// The opacity table has `inf` where the opacity is not defined (rho < 3.4e-7
/// g/cm^3 and T = 3e4 to 5e6 K, see `input_data/README.md`). It is kept as is,
/// so the opacity there is `inf` or `nan`.
template <typename Real> struct TableEOS {
  /// @brief ln(T) and ln(kappa) on the (ln rho, ln e) grid
  Array2D<Real> log_te, log_op;
  /// @brief Range of the axes
  Real log_ro_min, log_ro_max, log_en_min, log_en_max;

  explicit TableEOS(const fs::path &path) : TableEOS(TableFile(path)) {}

  explicit TableEOS(const TableFile &file)
      : log_te(shape_of(file, "ro"), shape_of(file, "en")),
        log_op(shape_of(file, "ro"), shape_of(file, "en")) {
    if (file.kind != "eos_table_roen") {
      throw std::runtime_error("Expected a table of kind eos_table_roen, got: " +
                               file.kind);
    }
    const int n_ro = log_te.extent(0);
    const int n_en = log_te.extent(1);
    const auto &ro = file.get("ro", {n_ro});
    const auto &en = file.get("en", {n_en});
    const auto &te = file.get("te", {n_ro, n_en});
    const auto &op = file.get("op", {n_ro, n_en});

    log_ro_min = static_cast<Real>(ro.front());
    log_ro_max = static_cast<Real>(ro.back());
    log_en_min = static_cast<Real>(en.front());
    log_en_max = static_cast<Real>(en.back());

    // Cells where the EOS failed have te = 0. A floor keeps the logarithm
    // finite. The same floor is used for the opacity.
    constexpr double floor = 1.e-20;
    for (int i = 0; i < n_ro; ++i) {
      for (int j = 0; j < n_en; ++j) {
        const int ij = i * n_en + j;
        log_te(i, j) = static_cast<Real>(std::log(std::max(te[ij], floor)));
        log_op(i, j) = static_cast<Real>(std::log(std::max(op[ij], floor)));
      }
    }
  }

  /// @brief Temperature [K] at density ro [g/cm^3] and specific internal
  /// energy ei [erg/g]
  Real temperature(Real ro, Real ei) const {
    return std::exp(interpolate(log_te, ro, ei));
  }

  /// @brief Rosseland mean opacity [cm^2/g] at density ro [g/cm^3] and
  /// specific internal energy ei [erg/g]
  Real opacity(Real ro, Real ei) const {
    return std::exp(interpolate(log_op, ro, ei));
  }

private:
  static int shape_of(const TableFile &file, const std::string &name) {
    const auto it = file.shapes.find(name);
    if (it == file.shapes.end() || it->second.size() != 1) {
      throw std::runtime_error("Table has no 1D axis '" + name + "'");
    }
    return it->second[0];
  }

  Real interpolate(const Array2D<Real> &table, Real ro, Real ei) const {
    const int n_ro = table.extent(0);
    const int n_en = table.extent(1);
    const Real dro_inv = Real(n_ro - 1) / (log_ro_max - log_ro_min);
    const Real den_inv = Real(n_en - 1) / (log_en_max - log_en_min);
    return interpolate_uniform_table2d(table.data(), log_ro_min, dro_inv,
                                       log_en_min, den_inv, std::log(ro),
                                       std::log(ei), n_ro, n_en);
  }
};

}  // namespace eos
}  // namespace miso
