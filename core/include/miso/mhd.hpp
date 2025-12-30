#pragma once

#include <miso/array3d.hpp>
#include <miso/env.hpp>
#include <miso/eos.hpp>
#include <miso/grid.hpp>
#include <miso/mhd_fields.hpp>
#include <miso/mhd_halo_exchange.hpp>
#include <miso/mhd_integrator.hpp>
#include <miso/time.hpp>

namespace miso {
namespace mhd {

template <typename Real, typename BoundaryCondition, typename EOS,
          typename Source>
struct MHD {
  Time<Real> &time;
  Grid<Real, HostSpace> &grid;
#ifdef USE_CUDA
  Grid<Real, CUDASpace> grid_d;
#endif

  Fields<Real, HostSpace> qq;  // Required for cpu and gpu both
#ifdef USE_CUDA
  Fields<Real, CUDASpace> qq_d;
#endif

#ifdef USE_CUDA
  ExecContext<CUDASpace> &exec_ctx;
  impl_cuda::Integrator<Real, BoundaryCondition, EOS, Source> integrator;
#else
  ExecContext<HostSpace> &exec_ctx;
  impl_host::Integrator<Real, BoundaryCondition, EOS, Source> integrator;
#endif

  int n_output_digits;
  std::string mhd_save_dir;

  template <typename ExecContextType>
  MHD(Config &config, Time<Real> &time, Grid<Real, HostSpace> &grid,
      ExecContextType &exec_ctx)
#ifdef USE_CUDA
      : time(time), grid(grid), grid_d(grid), qq(grid), qq_d(grid_d),
        exec_ctx(exec_ctx), integrator(config, qq_d, grid_d, exec_ctx)
#else
      : time(time), grid(grid), qq(grid), exec_ctx(exec_ctx),
        integrator(config, qq, grid, exec_ctx)
#endif
  {
    n_output_digits = config["mhd"]["n_output_digits"].template as<int>();
    mhd_save_dir = config.save_dir +
                   config["mhd"]["mhd_save_dir"].template as<std::string>();
  }

  Real cfl() const { return integrator.cfl(); }

  void update(const Real dt) { integrator.update(dt); }

  void save() {
    util::create_directories(mhd_save_dir);
    std::string filename =
        mhd_save_dir + "mhd." + util::zfill(time.n_output, time.n_output_digits) +
        "." + util::zfill(mpi::rank(), n_output_digits) + ".bin";
    std::ofstream ofs(filename, std::ios::binary);
    assert(ofs.is_open());

    auto write_array = [&ofs](const Array3D<Real, HostSpace> &arr) {
      ofs.write(reinterpret_cast<const char *>(arr.data()),
                sizeof(Real) * arr.size());
    };
#ifdef USE_CUDA
    qq.copy_from(qq_d);
#endif  // USE_CUDA
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
    std::string filename =
        mhd_save_dir + "mhd." + util::zfill(time.n_output, time.n_output_digits) +
        "." + util::zfill(mpi::rank(), n_output_digits) + ".bin";
    std::ifstream ifs(filename, std::ios::binary);
    assert(ifs.is_open());

    auto read_array = [&ifs](Array3D<Real, HostSpace> &arr) {
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

}  // namespace mhd
}  // namespace miso
