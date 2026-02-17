#pragma once

#include "array3d.hpp"
#include "env.hpp"
#include "grid.hpp"
#include "mhd_fields.hpp"
#include "time.hpp"

namespace miso {
namespace mhd {

/// @brief Control reading and writing of MHD simulation checkpoint files
template <typename Real> struct Checkpoint {
  Fields<Real, backend::Host> qq;

  int n_output_digits;
  std::string mhd_save_dir;

  Checkpoint(Config &config, Grid<Real, backend::Host> &grid) : qq(grid) {
    n_output_digits = config["mhd"]["n_output_digits"].as<int>();
    mhd_save_dir =
        config.save_dir + config["mhd"]["mhd_save_dir"].as<std::string>();
  }

  std::string get_filename(const Time<Real> &time) const {
    return mhd_save_dir + "mhd." +
           util::zfill(time.n_output, time.n_output_digits) + "." +
           util::zfill(mpi::rank(), n_output_digits) + ".bin";
  }

  template <typename Backend>
  void save(const Time<Real> &time, const Fields<Real, Backend> &qq_) {
    qq.copy_from(qq_);

    util::create_directories(mhd_save_dir);
    std::string filename = get_filename(time);
    std::ofstream ofs(filename, std::ios::binary);
    assert(ofs.is_open());

    constexpr std::uint32_t elem_size = sizeof(Real);
    ofs.write(reinterpret_cast<const char *>(&elem_size), sizeof(std::uint32_t));

    auto write_array = [&ofs](const Array3D<Real, backend::Host> &arr) {
      ofs.write(reinterpret_cast<const char *>(arr.data()),
                sizeof(Real) * arr.size());
    };
    write_array(qq.ro);
    write_array(qq.vx);
    write_array(qq.vy);
    write_array(qq.vz);
    write_array(qq.bx);
    write_array(qq.by);
    write_array(qq.bz);
    write_array(qq.ei);
    write_array(qq.ph);
  };

  template <typename Backend>
  void load(const Time<Real> &time, Fields<Real, Backend> &qq_) {
    std::string filename = get_filename(time);
    std::ifstream ifs(filename, std::ios::binary);
    assert(ifs.is_open());

    std::uint32_t elem_size;
    ifs.read(reinterpret_cast<char *>(&elem_size), sizeof(std::uint32_t));
    if (elem_size != sizeof(Real)) {
      throw std::runtime_error(
          "Checkpoint file element size does not match Real type size.");
    }

    auto read_array = [&ifs](Array3D<Real, backend::Host> &arr) {
      ifs.read(reinterpret_cast<char *>(arr.data()), sizeof(Real) * arr.size());
    };
    read_array(qq.ro);
    read_array(qq.vx);
    read_array(qq.vy);
    read_array(qq.vz);
    read_array(qq.bx);
    read_array(qq.by);
    read_array(qq.bz);
    read_array(qq.ei);
    read_array(qq.ph);

    qq_.copy_from(qq);
  };
};

}  // namespace mhd
}  // namespace miso
