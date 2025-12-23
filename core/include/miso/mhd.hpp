#pragma once

#include <miso/eos.hpp>

#include <miso/mhd_cpu.hpp>
#ifdef USE_CUDA
#include <miso/mhd_gpu.cuh>
#endif

namespace miso {
namespace mhd {

template <typename Real, typename BoundaryCondition, typename EOS,
          typename Source>
struct MHD {
  Config &config;
  Time<Real> &time;
  Grid<Real> &grid;

  cpu::Fields<Real> qq;  // Required for cpu and gpu both
#ifdef USE_CUDA
  gpu::Fields<Real> qq_d;
#endif

#ifdef USE_CUDA
  gpu::HaloExchanger<Real> halo_exchanger;
  gpu::Integrator<Real, BoundaryCondition, EOS, Source> integrator;
#else
  cpu::HaloExchanger<Real> halo_exchanger;
  cpu::Integrator<Real, BoundaryCondition, EOS, Source> integrator;
#endif

#ifdef USE_CUDA
  MHD(Config &config, Time<Real> &time, Grid<Real> &grid)
      : config(config), time(time), grid(grid), qq(grid), qq_d(grid),
        integrator(mhd) {}
#else
  MHD(Config &config, Time<Real> &time, Grid<Real> &grid)
      : config(config), time(time), grid(grid), qq(grid), integrator(mhd) {}
#endif

  void save() const {
    const auto n_output_digits =
        config["mhd"]["n_output_digits"].template as<int>();
    const auto mhd_save_dir =
        config.save_dir +
        config["mhd"]["mhd_save_dir"].template as<std::string>();
    util::create_directories(mhd_save_dir);

    std::string filename =
        mhd_save_dir + "mhd." + util::zfill(time.n_output, time.n_output_digits) +
        "." + util::zfill(mpi::rank(), n_output_digits) + ".bin";
    std::ofstream ofs(filename, std::ios::binary);
    assert(ofs.is_open());

    auto write_array = [&ofs](const Array3D<Real> &arr) {
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

  void load() {
    const auto n_output_digits =
        config["mhd"]["n_output_digits"].template as<int>();
    const auto mhd_save_dir =
        config.save_dir +
        config["mhd"]["mhd_save_dir"].template as<std::string>();
    std::string filename =
        mhd_save_dir + "mhd." + util::zfill(time.n_output, time.n_output_digits) +
        "." + util::zfill(mpi::rank(), n_output_digits) + ".bin";
    std::ifstream ifs(filename, std::ios::binary);
    assert(ifs.is_open());

    auto read_array = [&ifs](Array3D<Real> &arr) {
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
  };
};

/// @brief Ideal MHD model type alias
template <typename Real, typename BoundaryCondition>
using IdealMHD =
    MHD<Real, BoundaryCondition, eos::IdealEOS<Real>, NoSource<Real>>;

}  // namespace mhd
}  // namespace miso
