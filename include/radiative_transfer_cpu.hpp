///
/// @brief Radiative Transfer (RT) module for CPU
/// @details Implements a monochromatic RT solver for CPU-based simulations.
///
#pragma once

#include <cassert>
#include <vector>

#include "angular_quadrature.hpp"
#include "array3d_cpu.hpp"
#include "array4d_cpu.hpp"
#include "grid_cpu.hpp"
#include "mpi_manager.hpp"
#include "mpi_types.hpp"
#include "utility.hpp"

///
/// @brief Radiation transfer variables
///
template <typename Real> struct RTVars {
  /// @brief Radiation intensity
  Array4D<Real> rint, rint_old;

  /// @brief Source function
  Array3D<Real> src_func;

  /// @brief Absorption coefficient
  Array3D<Real> abs_coeff;

  /// @brief Angular quadrature
  AngularQuadrature<Real> ang_quad;

  RTCore(const Grid<Real> &grid, const int num_rays)
      : rint(num_rays, grid.i_total, grid.j_total, grid.k_total),
        rint_old(num_rays, grid.i_total, grid.j_total, grid.k_total),
        src_func(grid.i_total, grid.j_total, grid.k_total),
        abs_coeff(grid.i_total, grid.j_total, grid.k_total),
        ang_quad(grid.num_rays) {}
};

///
/// @brief Radiation transfer manager
///
template <typename Real> struct RTManager {
  /// @brief  Radiation transfer variables
  RTVars<Real> rt_vars;

  /// @brief  MPI communication buffers
  Array3D<Real> recv_buff_x_pos, recv_buff_x_neg;
  Array3D<Real> recv_buff_y_pos, recv_buff_y_neg;
  Array3D<Real> recv_buff_z_pos, recv_buff_z_neg;
  Array3D<Real> send_buff_x_pos, send_buff_x_neg;
  Array3D<Real> send_buff_y_pos, send_buff_y_neg;
  Array3D<Real> send_buff_z_pos, send_buff_z_neg;

  /// TODO: `num_rays` of buffers can be reduced considering ray directions.
  RTManager(const Grid<Real> &grid, const int num_rays)
      : rt_vars(grid, num_rays),
        recv_buff_x_pos(num_rays, grid.j_total, grid.k_total),
        recv_buff_x_neg(num_rays, grid.j_total, grid.k_total),
        recv_buff_y_pos(num_rays, grid.i_total, grid.k_total),
        recv_buff_y_neg(num_rays, grid.i_total, grid.k_total),
        recv_buff_z_pos(num_rays, grid.i_total, grid.j_total),
        recv_buff_z_neg(num_rays, grid.i_total, grid.j_total),
        send_buff_x_pos(num_rays, grid.j_total, grid.k_total),
        send_buff_x_neg(num_rays, grid.j_total, grid.k_total),
        send_buff_y_pos(num_rays, grid.i_total, grid.k_total),
        send_buff_y_neg(num_rays, grid.i_total, grid.k_total),
        send_buff_z_pos(num_rays, grid.i_total, grid.j_total),
        send_buff_z_neg(num_rays, grid.i_total, grid.j_total) {}

  void save(const std::string &file_path) const {
    std::ofstream ofs(file_path, std::ios::binary);
    assert(ofs.is_open());

    auto write_array1d = [&ofs](const std::vector<Real> &arr) {
      ofs.write(reinterpret_cast<const char *>(arr.data()),
                sizeof(Real) * arr.size());
    };
    auto write_array3d = [&ofs](const Array3D<Real> &arr) {
      ofs.write(reinterpret_cast<const char *>(arr.data()),
                sizeof(Real) * arr.size());
    };
    auto write_array4d = [&ofs](const Array4D<Real> &arr) {
      ofs.write(reinterpret_cast<const char *>(arr.data()),
                sizeof(Real) * arr.size());
    };

    ofs.write(reinterpret_cast<const char *>(&rt_vars.ang_quad.num_rays),
              sizeof(int));
    write_array1d(rt_vars.ang_quad.weights);
    write_array1d(rt_vars.ang_quad.mu_x);
    write_array1d(rt_vars.ang_quad.mu_y);
    write_array1d(rt_vars.ang_quad.mu_z);
    write_array3d(rt_vars.src_func);
    write_array3d(rt_vars.abs_coeff);
    write_array4d(rt_vars.rint);
    ofs.close();
  };

  void load(const std::string &file_path) {
    std::ifstream ifs(file_path, std::ios::binary);
    assert(ifs.is_open());

    auto read_array1d = [&ifs](std::vector<Real> &arr) {
      ifs.read(reinterpret_cast<char *>(arr.data()), sizeof(Real) * arr.size());
    };
    auto read_array3d = [&ifs](Array3D<Real> &arr) {
      ifs.read(reinterpret_cast<char *>(arr.data()), sizeof(Real) * arr.size());
    };
    auto read_array4d = [&ifs](Array4d<Real> &arr) {
      ifs.read(reinterpret_cast<char *>(arr.data()), sizeof(Real) * arr.size());
    };

    int num_rays;
    ifs.read(reinterpret_cast<char *>(&num_rays), sizeof(int));
    assert(num_rays == rt_vars.ang_quad.num_rays);
    read_array1d(rt_vars.ang_quad.weights);
    read_array1d(rt_vars.ang_quad.mu_x);
    read_array1d(rt_vars.ang_quad.mu_y);
    read_array1d(rt_vars.ang_quad.mu_z);
    read_array3d(rt_vars.src_func);
    read_array3d(rt_vars.abs_coeff);
    read_array4d(rt_vars.rint);
    ifs.close();
  };

  /// @brief Exchange halo data between MPI processes
  void mpi_exchange_halo_x(Grid<Real> &grid, MPIManager<Real> &mpi_manager) {
    mpi_exchange_halo_z(grid, mpi_manager);
    mpi_exchange_halo_y(grid, mpi_manager);
    mpi_exchange_halo_x(grid, mpi_manager);
  }

  /// @brief Exchange halo data between MPI processes
  void mpi_exchange_halo_x(Grid<Real> &grid, MPIManager<Real> &mpi_manager) {
    MPI_Request reqs[4];
    int req_count = 0;

    // left/right indices
    // * margin = 1; ks = 1; kb0 = 0;
    // * k_size = k_total - margin*2;
    // * kb1 = k_total - 1;
    // * The right-most grid (i.e., kb1) is not used in radiative transfer.
    const auto ib0 = grid.margin - grid.is;
    const auto ib1 = grid.ib0 + grid.i_size + grid.is;
    const auto jb0 = grid.margin - grid.js;
    const auto jb1 = grid.jb0 + grid.j_size + grid.js;
    const auto kb0 = grid.margin - grid.ks;
    const auto kb1 = grid.kb0 + grid.k_size + grid.ks;

    // positive x-direction
    if (mpi_manager.x_procs_pos != MPI_PROC_NULL && grid.i_size > 1) {
      for (int i_ray = 0; i_ray < rt_vars.ang_quad.num_rays; ++i_ray) {
        if (rt_vars.ang_quad.mu_x[i_ray] > 0.0) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              send_buff_x_pos(i_ray, j, k) = rt_vars.rint(i_ray, ib1 - 1, j, k);
            }
          }
        }
      }
      MPI_Isend(send_buff_x_pos.data(), send_buff_x_pos.size(), mpi_type<Real>(),
                mpi_manager.x_procs_pos, 1100, mpi_manager.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(recv_buff_x_pos.data(), recv_buff_x_pos.size(), mpi_type<Real>(),
                mpi_manager.x_procs_pos, 1200, mpi_manager.cart_comm,
                &reqs[req_count++]);
    }

    // negative x-direction
    if (mpi_manager.x_procs_neg != MPI_PROC_NULL && grid.i_size > 1) {
      for (int i_ray = 0; i_ray < rt_vars.ang_quad.num_rays; ++i_ray) {
        if (rt_vars.ang_quad.mu_x[i_ray] < 0.0) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              send_buff_x_neg(i_ray, j, k) = rt_vars.rint(i_ray, ib0, j, k);
            }
          }
        }
      }
      MPI_Isend(send_buff_x_neg.data(), send_buff_x_neg.size(), mpi_type<Real>(),
                mpi_manager.x_procs_neg, 1200, mpi_manager.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(recv_buff_x_neg.data(), recv_buff_x_neg.size(), mpi_type<Real>(),
                mpi_manager.x_procs_neg, 1100, mpi_manager.cart_comm,
                &reqs[req_count++]);
    }

    if (req_count > 0) {
      MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    }

    // ################
    // x_pos direction
    if (mpi_manager.x_procs_pos != MPI_PROC_NULL && grid.i_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_margin; ++i) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              (*vars[v])(grid.i_total - grid.i_margin + i, j, k) =
                  recv_buff_x_pos(i, j, k, v);
            }
          }
        }
      }
    }

    // x_neg direction
    if (mpi_manager.x_procs_neg != MPI_PROC_NULL && grid.i_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_margin; ++i) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              (*vars[v])(i, j, k) = recv_buff_x_neg(i, j, k, v);
            }
          }
        }
      }
    }
  }
};
