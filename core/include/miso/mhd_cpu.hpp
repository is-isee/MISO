#pragma once

#include <cassert>
#include <vector>

#include <miso/array3d_cpu.hpp>
#include <miso/array4d_cpu.hpp>
#include <miso/grid_cpu.hpp>
#include <miso/mpi_manager.hpp>
#include <miso/mpi_types.hpp>
#include <miso/utility.hpp>

namespace miso {
namespace mhd {
namespace cpu {

template <typename Real> struct Fields {
  Array3D<Real> ro, vx, vy, vz, bx, by, bz, ei, ph;

  Fields(int i_size, int j_size, int k_size)
      : ro(i_size, j_size, k_size), vx(i_size, j_size, k_size),
        vy(i_size, j_size, k_size), vz(i_size, j_size, k_size),
        bx(i_size, j_size, k_size), by(i_size, j_size, k_size),
        bz(i_size, j_size, k_size), ei(i_size, j_size, k_size),
        ph(i_size, j_size, k_size) {}

  void copy_from(const Fields &other) {
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

template <typename Real> struct HaloExchanger {
  /// @brief  MPI communication buffers
  Array4D<Real> recv_x_pos, recv_x_neg;
  Array4D<Real> recv_y_pos, recv_y_neg;
  Array4D<Real> recv_z_pos, recv_z_neg;
  Array4D<Real> send_x_pos, send_x_neg;
  Array4D<Real> send_y_pos, send_y_neg;
  Array4D<Real> send_z_pos, send_z_neg;

  Grid<Real> &grid;

  HaloExchanger(Grid<Real> &grid)
      : recv_x_pos(grid.i_margin, grid.j_total, grid.k_total, 9),
        recv_x_neg(grid.i_margin, grid.j_total, grid.k_total, 9),
        recv_y_pos(grid.i_total, grid.j_margin, grid.k_total, 9),
        recv_y_neg(grid.i_total, grid.j_margin, grid.k_total, 9),
        recv_z_pos(grid.i_total, grid.j_total, grid.k_margin, 9),
        recv_z_neg(grid.i_total, grid.j_total, grid.k_margin, 9),
        send_x_pos(grid.i_margin, grid.j_total, grid.k_total, 9),
        send_x_neg(grid.i_margin, grid.j_total, grid.k_total, 9),
        send_y_pos(grid.i_total, grid.j_margin, grid.k_total, 9),
        send_y_neg(grid.i_total, grid.j_margin, grid.k_total, 9),
        send_z_pos(grid.i_total, grid.j_total, grid.k_margin, 9),
        send_z_neg(grid.i_total, grid.j_total, grid.k_margin, 9), grid(grid) {}

  void exchange(Fields<Real> &qq_trgt, MPIManager &mpi) {
    std::array<Array3D<Real> *, 9> vars = {&qq_trgt.ro, &qq_trgt.vx, &qq_trgt.vy,
                                           &qq_trgt.vz, &qq_trgt.bx, &qq_trgt.by,
                                           &qq_trgt.bz, &qq_trgt.ei, &qq_trgt.ph};
    MPI_Request reqs[12];
    int req_count = 0;

    // ################
    // positive x-direction
    if (mpi.x_procs_pos != MPI_PROC_NULL && grid.i_size > 1) {
      for (int i = 0; i < grid.i_margin; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          for (int k = 0; k < grid.k_total; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_x_pos(i, j, k, v) =
                  (*vars[v])(grid.i_total - 2 * grid.i_margin + i, j, k);
            }
          }
        }
      }
      MPI_Isend(send_x_pos.data(), send_x_pos.size(), mpi_type<Real>(),
                mpi.x_procs_pos, 100, mpi.cart_comm, &reqs[req_count++]);
      MPI_Irecv(recv_x_pos.data(), recv_x_pos.size(), mpi_type<Real>(),
                mpi.x_procs_pos, 200, mpi.cart_comm, &reqs[req_count++]);
    }

    // negative x-direction
    if (mpi.x_procs_neg != MPI_PROC_NULL && grid.i_size > 1) {
      for (int i = 0; i < grid.i_margin; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          for (int k = 0; k < grid.k_total; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_x_neg(i, j, k, v) = (*vars[v])(grid.i_margin + i, j, k);
            }
          }
        }
      }
      MPI_Isend(send_x_neg.data(), send_x_neg.size(), mpi_type<Real>(),
                mpi.x_procs_neg, 200, mpi.cart_comm, &reqs[req_count++]);
      MPI_Irecv(recv_x_neg.data(), recv_x_neg.size(), mpi_type<Real>(),
                mpi.x_procs_neg, 100, mpi.cart_comm, &reqs[req_count++]);
    }

    // ################
    // positive y-direction
    if (mpi.y_procs_pos != MPI_PROC_NULL && grid.j_size > 1) {
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_margin; ++j) {
          for (int k = 0; k < grid.k_total; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_y_pos(i, j, k, v) =
                  (*vars[v])(i, grid.j_total - 2 * grid.j_margin + j, k);
            }
          }
        }
      }
      MPI_Isend(send_y_pos.data(), send_y_pos.size(), mpi_type<Real>(),
                mpi.y_procs_pos, 300, mpi.cart_comm, &reqs[req_count++]);
      MPI_Irecv(recv_y_pos.data(), recv_y_pos.size(), mpi_type<Real>(),
                mpi.y_procs_pos, 400, mpi.cart_comm, &reqs[req_count++]);
    }

    // negative y-direction
    if (mpi.y_procs_neg != MPI_PROC_NULL && grid.j_size > 1) {
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_margin; ++j) {
          for (int k = 0; k < grid.k_total; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_y_neg(i, j, k, v) = (*vars[v])(i, grid.j_margin + j, k);
            }
          }
        }
      }
      MPI_Isend(send_y_neg.data(), send_y_neg.size(), mpi_type<Real>(),
                mpi.y_procs_neg, 400, mpi.cart_comm, &reqs[req_count++]);
      MPI_Irecv(recv_y_neg.data(), recv_y_neg.size(), mpi_type<Real>(),
                mpi.y_procs_neg, 300, mpi.cart_comm, &reqs[req_count++]);
    }

    // ################
    // positive z-direction
    if (mpi.z_procs_pos != MPI_PROC_NULL && grid.k_size > 1) {
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          for (int k = 0; k < grid.k_margin; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_z_pos(i, j, k, v) =
                  (*vars[v])(i, j, grid.k_total - 2 * grid.k_margin + k);
            }
          }
        }
      }
      MPI_Isend(send_z_pos.data(), send_z_pos.size(), mpi_type<Real>(),
                mpi.z_procs_pos, 500, mpi.cart_comm, &reqs[req_count++]);
      MPI_Irecv(recv_z_pos.data(), recv_z_pos.size(), mpi_type<Real>(),
                mpi.z_procs_pos, 600, mpi.cart_comm, &reqs[req_count++]);
    }

    // negative z-direction
    if (mpi.z_procs_neg != MPI_PROC_NULL && grid.k_size > 1) {
      for (int i = 0; i < grid.i_total; ++i) {
        for (int j = 0; j < grid.j_total; ++j) {
          for (int k = 0; k < grid.k_margin; ++k) {
            for (int v = 0; v < 9; ++v) {
              send_z_neg(i, j, k, v) = (*vars[v])(i, j, grid.k_margin + k);
            }
          }
        }
      }
      MPI_Isend(send_z_neg.data(), send_z_neg.size(), mpi_type<Real>(),
                mpi.z_procs_neg, 600, mpi.cart_comm, &reqs[req_count++]);
      MPI_Irecv(recv_z_neg.data(), recv_z_neg.size(), mpi_type<Real>(),
                mpi.z_procs_neg, 500, mpi.cart_comm, &reqs[req_count++]);
    }

    if (req_count > 0) {
      MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    }

    // ################
    // positive x-direction
    if (mpi.x_procs_pos != MPI_PROC_NULL && grid.i_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_margin; ++i) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              (*vars[v])(grid.i_total - grid.i_margin + i, j, k) =
                  recv_x_pos(i, j, k, v);
            }
          }
        }
      }
    }

    // negative x-direction
    if (mpi.x_procs_neg != MPI_PROC_NULL && grid.i_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_margin; ++i) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              (*vars[v])(i, j, k) = recv_x_neg(i, j, k, v);
            }
          }
        }
      }
    }

    // ################
    // positive y-direction
    if (mpi.y_procs_pos != MPI_PROC_NULL && grid.j_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_total; ++i) {
          for (int j = 0; j < grid.j_margin; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              (*vars[v])(i, grid.j_total - grid.j_margin + j, k) =
                  recv_y_pos(i, j, k, v);
            }
          }
        }
      }
    }

    // negative y-direction
    if (mpi.y_procs_neg != MPI_PROC_NULL && grid.j_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_total; ++i) {
          for (int j = 0; j < grid.j_margin; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              (*vars[v])(i, j, k) = recv_y_neg(i, j, k, v);
            }
          }
        }
      }
    }

    // ################
    // positive z-direction
    if (mpi.z_procs_pos != MPI_PROC_NULL && grid.k_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_total; ++i) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_margin; ++k) {
              (*vars[v])(i, j, grid.k_total - grid.k_margin + k) =
                  recv_z_pos(i, j, k, v);
            }
          }
        }
      }
    }

    // negative z-direction
    if (mpi.z_procs_neg != MPI_PROC_NULL && grid.k_size > 1) {
      for (int v = 0; v < 9; ++v) {
        for (int i = 0; i < grid.i_total; ++i) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_margin; ++k) {
              (*vars[v])(i, j, k) = recv_z_neg(i, j, k, v);
            }
          }
        }
      }
    }
  }
};

}  // namespace cpu
}  // namespace mhd
}  // namespace miso
