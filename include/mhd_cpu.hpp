#pragma once

#include <cassert>
#include <vector>

#include "array3d_cpu.hpp"
#include "array4d.hpp"
#include "grid_cpu.hpp"
#include "mpi_manager.hpp"
#include "mpi_types.hpp"
#include "time_cpu.hpp"
#include "utility.hpp"

template <typename Real> struct MHDCore {
  Array3D<Real> ro, vx, vy, vz, bx, by, bz, ei, ph;

  MHDCore(int i_size, int j_size, int k_size)
      : ro(i_size, j_size, k_size), vx(i_size, j_size, k_size),
        vy(i_size, j_size, k_size), vz(i_size, j_size, k_size),
        bx(i_size, j_size, k_size), by(i_size, j_size, k_size),
        bz(i_size, j_size, k_size), ei(i_size, j_size, k_size),
        ph(i_size, j_size, k_size) {}
  void copy_from(const MHDCore &other) {
    ro.copy_from(other.ro);
    vx.copy_from(other.vx);
    vy.copy_from(other.vy);
    vz.copy_from(other.vz);
    bx.copy_from(other.bx);
    by.copy_from(other.by);
    bz.copy_from(other.bz);
    ei.copy_from(other.ei);
    ph.copy_from(other.ph);
  }
};

template <typename Real> struct MHD {
  MHDCore<Real> qq, qq_argm, qq_rslt;
  Real cfl_number;

  /// @brief  MPI communication buffers
  Array4D<Real> recv_buff_x_plus, recv_buff_x_mnus;
  Array4D<Real> recv_buff_y_plus, recv_buff_y_mnus;
  Array4D<Real> recv_buff_z_plus, recv_buff_z_mnus;
  Array4D<Real> send_buff_x_plus, send_buff_x_mnus;
  Array4D<Real> send_buff_y_plus, send_buff_y_mnus;
  Array4D<Real> send_buff_z_plus, send_buff_z_mnus;

  MHD(const Grid<Real> &grid)
      : qq(grid.i_total, grid.j_total, grid.k_total),
        qq_argm(grid.i_total, grid.j_total, grid.k_total),
        qq_rslt(grid.i_total, grid.j_total, grid.k_total),
        recv_buff_x_plus(grid.i_margin, grid.j_total, grid.k_total, 9),
        recv_buff_x_mnus(grid.i_margin, grid.j_total, grid.k_total, 9),
        recv_buff_y_plus(grid.i_total, grid.j_margin, grid.k_total, 9),
        recv_buff_y_mnus(grid.i_total, grid.j_margin, grid.k_total, 9),
        recv_buff_z_plus(grid.i_total, grid.j_total, grid.k_margin, 9),
        recv_buff_z_mnus(grid.i_total, grid.j_total, grid.k_margin, 9),
        send_buff_x_plus(grid.i_margin, grid.j_total, grid.k_total, 9),
        send_buff_x_mnus(grid.i_margin, grid.j_total, grid.k_total, 9),
        send_buff_y_plus(grid.i_total, grid.j_margin, grid.k_total, 9),
        send_buff_y_mnus(grid.i_total, grid.j_margin, grid.k_total, 9),
        send_buff_z_plus(grid.i_total, grid.j_total, grid.k_margin, 9),
        send_buff_z_mnus(grid.i_total, grid.j_total, grid.k_margin, 9) {}

  void save(const Config &config, const Time<Real> &time) const {
    std::string filename =
        config.mhd_save_dir + "mhd." +
        util::zfill(time.n_output, time.n_output_digits) + "." +
        util::zfill(config.mpi.myrank, config.mpi.n_procs_digits) + ".bin";
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
    ofs.close();
  };

  void load(const Config &config, const Time<Real> &time) {
    std::string filename =
        config.mhd_save_dir + "mhd." +
        util::zfill(time.n_output, time.n_output_digits) + "." +
        util::zfill(config.mpi.myrank, config.mpi.n_procs_digits) + ".bin";
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
    ifs.close();
  };

  void mpi_exchange_halo(MHDCore<Real> &qq_trgt, Grid<Real> &grid,
                         MPIManager<Real> &mpi) {
    std::array<Array3D<Real> *, 9> vars = {&qq_trgt.ro, &qq_trgt.vx, &qq_trgt.vy,
                                           &qq_trgt.vz, &qq_trgt.bx, &qq_trgt.by,
                                           &qq_trgt.bz, &qq_trgt.ei, &qq_trgt.ph};

    MPI_Request reqs[12];
    int req_count = 0;

    // ################
    // x_plus direction
    if (mpi.x_procs_plus != MPI_PROC_NULL && grid.i_size > 1) {
      for (int i = 0; i < grid.i_margin; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          for (int k = 0; k < grid.k_total; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_buff_x_plus(i, j, k, v) =
                  (*vars[v])(grid.i_total - 2 * grid.i_margin + i, j, k);
            }
          }
        }
      }
      MPI_Isend(send_buff_x_plus.data(), send_buff_x_plus.size(),
                mpi_type<Real>(), mpi.x_procs_plus, 100, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(recv_buff_x_plus.data(), recv_buff_x_plus.size(),
                mpi_type<Real>(), mpi.x_procs_plus, 200, mpi.cart_comm,
                &reqs[req_count++]);
    }

    // x_mnus direction
    if (mpi.x_procs_mnus != MPI_PROC_NULL && grid.i_size > 1) {
      for (int i = 0; i < grid.i_margin; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          for (int k = 0; k < grid.k_total; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_buff_x_mnus(i, j, k, v) = (*vars[v])(grid.i_margin + i, j, k);
            }
          }
        }
      }
      MPI_Isend(send_buff_x_mnus.data(), send_buff_x_mnus.size(),
                mpi_type<Real>(), mpi.x_procs_mnus, 200, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(recv_buff_x_mnus.data(), recv_buff_x_mnus.size(),
                mpi_type<Real>(), mpi.x_procs_mnus, 100, mpi.cart_comm,
                &reqs[req_count++]);
    }

    // ################
    // y_plus direction
    if (mpi.y_procs_plus != MPI_PROC_NULL && grid.j_size > 1) {
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_margin; ++j) {
          for (int k = 0; k < grid.k_total; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_buff_y_plus(i, j, k, v) =
                  (*vars[v])(i, grid.j_total - 2 * grid.j_margin + j, k);
            }
          }
        }
      }
      MPI_Isend(send_buff_y_plus.data(), send_buff_y_plus.size(),
                mpi_type<Real>(), mpi.y_procs_plus, 300, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(recv_buff_y_plus.data(), recv_buff_y_plus.size(),
                mpi_type<Real>(), mpi.y_procs_plus, 400, mpi.cart_comm,
                &reqs[req_count++]);
    }

    // y_mnus direction
    if (mpi.y_procs_mnus != MPI_PROC_NULL && grid.j_size > 1) {
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_margin; ++j) {
          for (int k = 0; k < grid.k_total; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_buff_y_mnus(i, j, k, v) = (*vars[v])(i, grid.j_margin + j, k);
            }
          }
        }
      }
      MPI_Isend(send_buff_y_mnus.data(), send_buff_y_mnus.size(),
                mpi_type<Real>(), mpi.y_procs_mnus, 400, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(recv_buff_y_mnus.data(), recv_buff_y_mnus.size(),
                mpi_type<Real>(), mpi.y_procs_mnus, 300, mpi.cart_comm,
                &reqs[req_count++]);
    }

    // ################
    // z_plus direction
    if (mpi.z_procs_plus != MPI_PROC_NULL && grid.k_size > 1) {
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          for (int k = 0; k < grid.k_margin; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_buff_z_plus(i, j, k, v) =
                  (*vars[v])(i, j, grid.k_total - 2 * grid.k_margin + k);
            }
          }
        }
      }
      MPI_Isend(send_buff_z_plus.data(), send_buff_z_plus.size(),
                mpi_type<Real>(), mpi.z_procs_plus, 500, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(recv_buff_z_plus.data(), recv_buff_z_plus.size(),
                mpi_type<Real>(), mpi.z_procs_plus, 600, mpi.cart_comm,
                &reqs[req_count++]);
    }

    // z_mnus direction
    if (mpi.z_procs_mnus != MPI_PROC_NULL && grid.k_size > 1) {
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          for (int k = 0; k < grid.k_margin; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_buff_z_mnus(i, j, k, v) = (*vars[v])(i, j, grid.k_margin + k);
            }
          }
        }
      }
      MPI_Isend(send_buff_z_mnus.data(), send_buff_z_mnus.size(),
                mpi_type<Real>(), mpi.z_procs_mnus, 600, mpi.cart_comm,
                &reqs[req_count++]);
      MPI_Irecv(recv_buff_z_mnus.data(), recv_buff_z_mnus.size(),
                mpi_type<Real>(), mpi.z_procs_mnus, 500, mpi.cart_comm,
                &reqs[req_count++]);
    }

    if (req_count > 0) {
      MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    }

    // ################
    // x_plus direction
    if (mpi.x_procs_plus != MPI_PROC_NULL && grid.i_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_margin; ++i) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              (*vars[v])(grid.i_total - grid.i_margin + i, j, k) =
                  recv_buff_x_plus(i, j, k, v);
            }
          }
        }
      }
    }

    // x_mnus direction
    if (mpi.x_procs_mnus != MPI_PROC_NULL && grid.i_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_margin; ++i) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              (*vars[v])(i, j, k) = recv_buff_x_mnus(i, j, k, v);
            }
          }
        }
      }
    }

    // ################
    // y_plus direction
    if (mpi.y_procs_plus != MPI_PROC_NULL && grid.j_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_total; ++i) {
          for (int j = 0; j < grid.j_margin; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              (*vars[v])(i, grid.j_total - grid.j_margin + j, k) =
                  recv_buff_y_plus(i, j, k, v);
            }
          }
        }
      }
    }

    // y_mnus direction
    if (mpi.y_procs_mnus != MPI_PROC_NULL && grid.j_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_total; ++i) {
          for (int j = 0; j < grid.j_margin; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              (*vars[v])(i, j, k) = recv_buff_y_mnus(i, j, k, v);
            }
          }
        }
      }
    }

    // ################
    // z_plus direction
    if (mpi.z_procs_plus != MPI_PROC_NULL && grid.k_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_total; ++i) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_margin; ++k) {
              (*vars[v])(i, j, grid.k_total - grid.k_margin + k) =
                  recv_buff_z_plus(i, j, k, v);
            }
          }
        }
      }
    }

    // z_mnus direction
    if (mpi.z_procs_mnus != MPI_PROC_NULL && grid.k_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_total; ++i) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_margin; ++k) {
              (*vars[v])(i, j, k) = recv_buff_z_mnus(i, j, k, v);
            }
          }
        }
      }
    }
  }
};