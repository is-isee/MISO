///
/// @brief Radiative Transfer (RT) module for CPU
/// @details Implements a monochromatic RT solver for CPU-based simulations.
///
#pragma once

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "array3d.hpp"
#include "array4d.hpp"
#include "boundary_condition.hpp"
#include "env.hpp"
#include "grid.hpp"
#include "mpi_util.hpp"
#include "rt_quadrature.hpp"
#include "utility.hpp"

namespace miso {
namespace rt {

template <typename Real> struct RT;

template <typename Real>
inline Real ray_cosine(const AngularQuadrature<Real> &ang_quad, int i_ray,
                       Direction direction) {
  switch (direction) {
  case Direction::X:
    return ang_quad.mu_x[i_ray];
  case Direction::Y:
    return ang_quad.mu_y[i_ray];
  case Direction::Z:
    return ang_quad.mu_z[i_ray];
  }
  return Real(0);
}

template <typename Real>
inline bool is_incoming_ray(const AngularQuadrature<Real> &ang_quad, int i_ray,
                            Direction direction, Side side) {
  const Real mu = ray_cosine(ang_quad, i_ray, direction);
  return (side == Side::INNER) ? (mu > Real(0)) : (mu < Real(0));
}

template <typename Real, typename ValueFn>
void set_incoming_boundary(RT<Real> &rt, const Grid<Real> &grid,
                           Direction direction, Side side, ValueFn &&value_fn) {
  (void)grid; // grid is intentionally unused in this overload
  int i_begin = rt.ib0;
  int i_end = rt.ib1;
  int j_begin = rt.jb0;
  int j_end = rt.jb1;
  int k_begin = rt.kb0;
  int k_end = rt.kb1;

  switch (direction) {
  case Direction::X:
    i_begin = (side == Side::INNER) ? rt.ib0 : rt.ib1;
    i_end = i_begin;
    break;
  case Direction::Y:
    j_begin = (side == Side::INNER) ? rt.jb0 : rt.jb1;
    j_end = j_begin;
    break;
  case Direction::Z:
    k_begin = (side == Side::INNER) ? rt.kb0 : rt.kb1;
    k_end = k_begin;
    break;
  }

  for (int i_ray = 0; i_ray < rt.ang_quad.num_rays; ++i_ray) {
    if (!is_incoming_ray(rt.ang_quad, i_ray, direction, side)) {
      continue;
    }
    for (int i = i_begin; i <= i_end; ++i) {
      for (int j = j_begin; j <= j_end; ++j) {
        for (int k = k_begin; k <= k_end; ++k) {
          rt.rint(i_ray, i, j, k) = value_fn(i, j, k);
        }
      }
    }
  }
}

template <typename Real, typename ValueFn>
void set_incoming_boundary_on_physical_faces(RT<Real> &rt, const Grid<Real> &grid,
                                             const mpi::Shape &mpi_shape,
                                             Direction direction, Side side,
                                             ValueFn &&value_fn) {
  namespace bc = miso::boundary_condition;
  if (!bc::is_physical_boundary(direction, side, mpi_shape)) {
    return;
  }
  set_incoming_boundary(rt, grid, direction, side,
                        std::forward<ValueFn>(value_fn));
}

///
/// @brief Radiation transfer solver
///
template <typename Real> struct RT {
  /// @brief Radiation intensity
  Array4D<Real> rint, rint_old;

  /// @brief Source function
  Array3D<Real> src_func;

  /// @brief Absorption coefficient
  Array3D<Real> abs_coeff;

  /// @brief Angular quadrature
  AngularQuadrature<Real> ang_quad;

  /// @brief MPI communication buffers
  Array3D<Real> recv_buff_x_pos, recv_buff_x_neg;
  Array3D<Real> recv_buff_y_pos, recv_buff_y_neg;
  Array3D<Real> recv_buff_z_pos, recv_buff_z_neg;
  Array3D<Real> send_buff_x_pos, send_buff_x_neg;
  Array3D<Real> send_buff_y_pos, send_buff_y_neg;
  Array3D<Real> send_buff_z_pos, send_buff_z_neg;

  /// @brief Left/right indices
  // * margin = 1; ks = 1; kb0 = 0;
  // * k_size = k_total - margin*2;
  // * kb1 = k_total - 2;
  // * The right-most grid (i.e., k=k_total-1) is not used in RT.
  const int ib0, ib1, jb0, jb1, kb0, kb1;

  /// TODO: `num_rays` of buffers can be reduced considering ray directions.
  RT(const Grid<Real> &grid, const int num_rays)
      : rint(num_rays, grid.i_total, grid.j_total, grid.k_total),
        rint_old(num_rays, grid.i_total, grid.j_total, grid.k_total),
        src_func(grid.i_total, grid.j_total, grid.k_total),
        abs_coeff(grid.i_total, grid.j_total, grid.k_total), ang_quad(num_rays),
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
        send_buff_z_neg(num_rays, grid.i_total, grid.j_total),
        ib0(grid.i_margin - grid.is), ib1(ib0 + grid.i_size),
        jb0(grid.j_margin - grid.js), jb1(jb0 + grid.j_size),
        kb0(grid.k_margin - grid.ks), kb1(kb0 + grid.k_size) {
    util::clear_array(rint);
    util::clear_array(rint_old);
    util::clear_array(src_func);
    util::clear_array(abs_coeff);
    util::clear_array(recv_buff_x_pos);
    util::clear_array(recv_buff_x_neg);
    util::clear_array(recv_buff_y_pos);
    util::clear_array(recv_buff_y_neg);
    util::clear_array(recv_buff_z_pos);
    util::clear_array(recv_buff_z_neg);
    util::clear_array(send_buff_x_pos);
    util::clear_array(send_buff_x_neg);
    util::clear_array(send_buff_y_pos);
    util::clear_array(send_buff_y_neg);
    util::clear_array(send_buff_z_pos);
    util::clear_array(send_buff_z_neg);
  }

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

    ofs.write(reinterpret_cast<const char *>(&ang_quad.num_rays), sizeof(int));
    write_array1d(ang_quad.weights);
    write_array1d(ang_quad.mu_x);
    write_array1d(ang_quad.mu_y);
    write_array1d(ang_quad.mu_z);
    write_array3d(src_func);
    write_array3d(abs_coeff);
    write_array4d(rint);
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
    auto read_array4d = [&ifs](Array4D<Real> &arr) {
      ifs.read(reinterpret_cast<char *>(arr.data()), sizeof(Real) * arr.size());
    };

    int num_rays;
    ifs.read(reinterpret_cast<char *>(&num_rays), sizeof(int));
    assert(num_rays == ang_quad.num_rays);
    read_array1d(ang_quad.weights);
    read_array1d(ang_quad.mu_x);
    read_array1d(ang_quad.mu_y);
    read_array1d(ang_quad.mu_z);
    read_array3d(src_func);
    read_array3d(abs_coeff);
    read_array4d(rint);
    ifs.close();
  };

  /// @brief Exchange halo data between MPI processes
  void mpi_exchange_halo(const Grid<Real> &grid, const mpi::Shape &mpi_shape) {
    mpi_exchange_halo_z(grid, mpi_shape);
    mpi_exchange_halo_y(grid, mpi_shape);
    mpi_exchange_halo_x(grid, mpi_shape);
  }

  /// @brief Exchange halo data (x-direction)
  void mpi_exchange_halo_x(const Grid<Real> &grid, const mpi::Shape &mpi_shape) {
    if (grid.i_size == 1) {
      return;
    }
    MPI_Request reqs[4];
    int req_count = 0;

    if (mpi_shape.x_procs_pos != MPI_PROC_NULL) {
      for (int i_ray = 0; i_ray < ang_quad.num_rays; ++i_ray) {
        if (ang_quad.mu_x[i_ray] >= 0.0) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              send_buff_x_pos(i_ray, j, k) = rint(i_ray, ib1, j, k);
            }
          }
        }
      }
      MPI_Isend(send_buff_x_pos.data(), send_buff_x_pos.size(),
                mpi::data_type<Real>(), mpi_shape.x_procs_pos, 1100,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(recv_buff_x_pos.data(), recv_buff_x_pos.size(),
                mpi::data_type<Real>(), mpi_shape.x_procs_pos, 1200,
                mpi_shape.cart_comm, &reqs[req_count++]);
    }

    if (mpi_shape.x_procs_neg != MPI_PROC_NULL) {
      for (int i_ray = 0; i_ray < ang_quad.num_rays; ++i_ray) {
        if (ang_quad.mu_x[i_ray] < 0.0) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              send_buff_x_neg(i_ray, j, k) = rint(i_ray, ib0, j, k);
            }
          }
        }
      }
      MPI_Isend(send_buff_x_neg.data(), send_buff_x_neg.size(),
                mpi::data_type<Real>(), mpi_shape.x_procs_neg, 1200,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(recv_buff_x_neg.data(), recv_buff_x_neg.size(),
                mpi::data_type<Real>(), mpi_shape.x_procs_neg, 1100,
                mpi_shape.cart_comm, &reqs[req_count++]);
    }

    if (req_count > 0) {
      MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    }

    if (mpi_shape.x_procs_pos != MPI_PROC_NULL) {
      for (int i_ray = 0; i_ray < ang_quad.num_rays; ++i_ray) {
        if (ang_quad.mu_x[i_ray] < 0.0) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              rint(i_ray, ib1, j, k) = recv_buff_x_pos(i_ray, j, k);
            }
          }
        }
      }
    }

    if (mpi_shape.x_procs_neg != MPI_PROC_NULL) {
      for (int i_ray = 0; i_ray < ang_quad.num_rays; ++i_ray) {
        if (ang_quad.mu_x[i_ray] >= 0.0) {
          for (int j = 0; j < grid.j_total; ++j) {
            for (int k = 0; k < grid.k_total; ++k) {
              rint(i_ray, ib0, j, k) = recv_buff_x_neg(i_ray, j, k);
            }
          }
        }
      }
    }
  }

  /// @brief Exchange halo data (y-direction)
  void mpi_exchange_halo_y(const Grid<Real> &grid, const mpi::Shape &mpi_shape) {
    if (grid.j_size == 1) {
      return;
    }
    MPI_Request reqs[4];
    int req_count = 0;

    if (mpi_shape.y_procs_pos != MPI_PROC_NULL) {
      for (int i_ray = 0; i_ray < ang_quad.num_rays; ++i_ray) {
        if (ang_quad.mu_y[i_ray] >= 0.0) {
          for (int i = 0; i < grid.i_total; ++i) {
            for (int k = 0; k < grid.k_total; ++k) {
              send_buff_y_pos(i_ray, i, k) = rint(i_ray, i, jb1, k);
            }
          }
        }
      }
      MPI_Isend(send_buff_y_pos.data(), send_buff_y_pos.size(),
                mpi::data_type<Real>(), mpi_shape.y_procs_pos, 1300,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(recv_buff_y_pos.data(), recv_buff_y_pos.size(),
                mpi::data_type<Real>(), mpi_shape.y_procs_pos, 1400,
                mpi_shape.cart_comm, &reqs[req_count++]);
    }

    if (mpi_shape.y_procs_neg != MPI_PROC_NULL) {
      for (int i_ray = 0; i_ray < ang_quad.num_rays; ++i_ray) {
        if (ang_quad.mu_y[i_ray] < 0.0) {
          for (int i = 0; i < grid.i_total; ++i) {
            for (int k = 0; k < grid.k_total; ++k) {
              send_buff_y_neg(i_ray, i, k) = rint(i_ray, i, jb0, k);
            }
          }
        }
      }
      MPI_Isend(send_buff_y_neg.data(), send_buff_y_neg.size(),
                mpi::data_type<Real>(), mpi_shape.y_procs_neg, 1400,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(recv_buff_y_neg.data(), recv_buff_y_neg.size(),
                mpi::data_type<Real>(), mpi_shape.y_procs_neg, 1300,
                mpi_shape.cart_comm, &reqs[req_count++]);
    }

    if (req_count > 0) {
      MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    }

    if (mpi_shape.y_procs_pos != MPI_PROC_NULL) {
      for (int i_ray = 0; i_ray < ang_quad.num_rays; ++i_ray) {
        if (ang_quad.mu_y[i_ray] < 0.0) {
          for (int i = 0; i < grid.i_total; ++i) {
            for (int k = 0; k < grid.k_total; ++k) {
              rint(i_ray, i, jb1, k) = recv_buff_y_pos(i_ray, i, k);
            }
          }
        }
      }
    }

    if (mpi_shape.y_procs_neg != MPI_PROC_NULL) {
      for (int i_ray = 0; i_ray < ang_quad.num_rays; ++i_ray) {
        if (ang_quad.mu_y[i_ray] >= 0.0) {
          for (int i = 0; i < grid.i_total; ++i) {
            for (int k = 0; k < grid.k_total; ++k) {
              rint(i_ray, i, jb0, k) = recv_buff_y_neg(i_ray, i, k);
            }
          }
        }
      }
    }
  }

  /// @brief Exchange halo data (z-direction)
  void mpi_exchange_halo_z(const Grid<Real> &grid, const mpi::Shape &mpi_shape) {
    if (grid.k_size == 1) {
      return;
    }
    MPI_Request reqs[4];
    int req_count = 0;

    if (mpi_shape.z_procs_pos != MPI_PROC_NULL) {
      for (int i_ray = 0; i_ray < ang_quad.num_rays; ++i_ray) {
        if (ang_quad.mu_z[i_ray] >= 0.0) {
          for (int i = 0; i < grid.i_total; ++i) {
            for (int j = 0; j < grid.j_total; ++j) {
              send_buff_z_pos(i_ray, i, j) = rint(i_ray, i, j, kb1);
            }
          }
        }
      }
      MPI_Isend(send_buff_z_pos.data(), send_buff_z_pos.size(),
                mpi::data_type<Real>(), mpi_shape.z_procs_pos, 1500,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(recv_buff_z_pos.data(), recv_buff_z_pos.size(),
                mpi::data_type<Real>(), mpi_shape.z_procs_pos, 1600,
                mpi_shape.cart_comm, &reqs[req_count++]);
    }

    if (mpi_shape.z_procs_neg != MPI_PROC_NULL) {
      for (int i_ray = 0; i_ray < ang_quad.num_rays; ++i_ray) {
        if (ang_quad.mu_z[i_ray] < 0.0) {
          for (int i = 0; i < grid.i_total; ++i) {
            for (int j = 0; j < grid.j_total; ++j) {
              send_buff_z_neg(i_ray, i, j) = rint(i_ray, i, j, kb0);
            }
          }
        }
      }
      MPI_Isend(send_buff_z_neg.data(), send_buff_z_neg.size(),
                mpi::data_type<Real>(), mpi_shape.z_procs_neg, 1600,
                mpi_shape.cart_comm, &reqs[req_count++]);
      MPI_Irecv(recv_buff_z_neg.data(), recv_buff_z_neg.size(),
                mpi::data_type<Real>(), mpi_shape.z_procs_neg, 1500,
                mpi_shape.cart_comm, &reqs[req_count++]);
    }

    if (req_count > 0) {
      MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    }

    if (mpi_shape.z_procs_pos != MPI_PROC_NULL) {
      for (int i_ray = 0; i_ray < ang_quad.num_rays; ++i_ray) {
        if (ang_quad.mu_z[i_ray] < 0.0) {
          for (int i = 0; i < grid.i_total; ++i) {
            for (int j = 0; j < grid.j_total; ++j) {
              rint(i_ray, i, j, kb1) = recv_buff_z_pos(i_ray, i, j);
            }
          }
        }
      }
    }

    if (mpi_shape.z_procs_neg != MPI_PROC_NULL) {
      for (int i_ray = 0; i_ray < ang_quad.num_rays; ++i_ray) {
        if (ang_quad.mu_z[i_ray] >= 0.0) {
          for (int i = 0; i < grid.i_total; ++i) {
            for (int j = 0; j < grid.j_total; ++j) {
              rint(i_ray, i, j, kb0) = recv_buff_z_neg(i_ray, i, j);
            }
          }
        }
      }
    }
  }

  /// @brief Sweep over the local grid by short characteristic method.
  /// @note  Currently, only uniform grid is supported.
  void single_sweep(const Grid<Real> &grid) {
    for (int i_ray = 0; i_ray < ang_quad.num_rays; ++i_ray) {
      const Real mu_x = ang_quad.mu_x[i_ray];
      const Real mu_y = ang_quad.mu_y[i_ray];
      const Real mu_z = ang_quad.mu_z[i_ray];

      // Coefficients for bilinear interpolation
      /// TODO: Assuming uniform grid. Should be generalized.
      static constexpr Real eps_mu = 1.e-30;
      const Real abs_mu_x = util::max2(std::abs(mu_x), eps_mu);
      const Real abs_mu_y = util::max2(std::abs(mu_y), eps_mu);
      const Real abs_mu_z = util::max2(std::abs(mu_z), eps_mu);
      const Real dx = (grid.i_size > 1) ? grid.dx[0] : 1.e30;
      const Real dy = (grid.j_size > 1) ? grid.dy[0] : 1.e30;
      const Real dz = (grid.k_size > 1) ? grid.dz[0] : 1.e30;
      const Real ds_x = dx / abs_mu_x;
      const Real ds_y = dy / abs_mu_y;
      const Real ds_z = dz / abs_mu_z;
      const Real ds = util::min3(ds_x, ds_y, ds_z);
      const Real cc_x = abs_mu_x * ds / dx;
      const Real cc_y = abs_mu_y * ds / dy;
      const Real cc_z = abs_mu_z * ds / dz;

      // Determine sweep order
      int ib, ie, it;
      if (mu_x >= 0.0) {
        ib = ib0 + grid.is;
        ie = ib1;
        it = 1;
      } else {
        ib = ib1 - grid.is;
        ie = ib0;
        it = -1;
      }
      int jb, je, jt;
      if (mu_y >= 0.0) {
        jb = jb0 + grid.js;
        je = jb1;
        jt = 1;
      } else {
        jb = jb1 - grid.js;
        je = jb0;
        jt = -1;
      }
      int kb, ke, kt;
      if (mu_z >= 0.0) {
        kb = kb0 + grid.ks;
        ke = kb1;
        kt = 1;
      } else {
        kb = kb1 - grid.ks;
        ke = kb0;
        kt = -1;
      }
      int ist = it * grid.is;
      int jst = jt * grid.js;
      int kst = kt * grid.ks;

      // Sweep over the local grid
      if ((ds_x <= ds_y) && (ds_x <= ds_z)) {  // yz-plane
        for (int i = ib; i != ie + it; i += it) {
          for (int j = util::min2(jb, je); j <= util::max2(jb, je); ++j) {
            for (int k = util::min2(kb, ke); k <= util::max2(kb, ke); ++k) {
              // Interpolate upwind values
              // clang-format off
              const Real rint_u = bilinear_interpolation(cc_y, cc_z,
                  rint(i_ray, i - ist, j      , k      ),
                  rint(i_ray, i - ist, j - jst, k      ),
                  rint(i_ray, i - ist, j      , k - kst),
                  rint(i_ray, i - ist, j - jst, k - kst));
              const Real src_func_u = bilinear_interpolation(cc_y, cc_z,
                  src_func(i - ist, j      , k      ),
                  src_func(i - ist, j - jst, k      ),
                  src_func(i - ist, j      , k - kst),
                  src_func(i - ist, j - jst, k - kst));
              const Real abs_coeff_u = bilinear_interpolation(cc_y, cc_z,
                  abs_coeff(i - ist, j      , k      ),
                  abs_coeff(i - ist, j - jst, k      ),
                  abs_coeff(i - ist, j      , k - kst),
                  abs_coeff(i - ist, j - jst, k - kst));
              // clang-format on

              // Integrate along a ray segment
              const Real dtau = integrate_optical_thickness(
                  abs_coeff_u, abs_coeff(i, j, k), ds);
              rint(i_ray, i, j, k) = integrate_radiative_intensity(
                  rint_u, src_func_u, src_func(i, j, k), dtau);
            }
          }
        }
      } else if ((ds_y <= ds_x) && (ds_y <= ds_z)) {  // xz-plane
        for (int j = jb; j != je + jt; j += jt) {
          for (int i = util::min2(ib, ie); i <= util::max2(ib, ie); ++i) {
            for (int k = util::min2(kb, ke); k <= util::max2(kb, ke); ++k) {
              // Interpolate upwind values
              // clang-format off
              const Real rint_u = bilinear_interpolation(cc_x, cc_z,
                  rint(i_ray, i      , j - jst, k      ),
                  rint(i_ray, i - ist, j - jst, k      ),
                  rint(i_ray, i      , j - jst, k - kst),
                  rint(i_ray, i - ist, j - jst, k - kst));
              const Real src_func_u = bilinear_interpolation(cc_x, cc_z,
                  src_func(i      , j - jst, k      ),
                  src_func(i - ist, j - jst, k      ),
                  src_func(i      , j - jst, k - kst),
                  src_func(i - ist, j - jst, k - kst));
              const Real abs_coeff_u = bilinear_interpolation(cc_x, cc_z,
                  abs_coeff(i      , j - jst, k      ),
                  abs_coeff(i - ist, j - jst, k      ),
                  abs_coeff(i      , j - jst, k - kst),
                  abs_coeff(i - ist, j - jst, k - kst));
              // clang-format on

              // Integrate along a ray segment
              const Real dtau = integrate_optical_thickness(
                  abs_coeff_u, abs_coeff(i, j, k), ds);
              rint(i_ray, i, j, k) = integrate_radiative_intensity(
                  rint_u, src_func_u, src_func(i, j, k), dtau);
            }
          }
        }
      } else {  // xy-plane
        for (int k = kb; k != ke + kt; k += kt) {
          for (int i = util::min2(ib, ie); i <= util::max2(ib, ie); ++i) {
            for (int j = util::min2(jb, je); j <= util::max2(jb, je); ++j) {
              // Interpolate upwind values
              // clang-format off
              const Real rint_u = bilinear_interpolation(cc_x, cc_y,
                  rint(i_ray, i      , j      , k - kst),
                  rint(i_ray, i - ist, j      , k - kst),
                  rint(i_ray, i      , j - jst, k - kst),
                  rint(i_ray, i - ist, j - jst, k - kst));
              const Real src_func_u = bilinear_interpolation(cc_x, cc_y,
                  src_func(i      , j      , k - kst),
                  src_func(i - ist, j      , k - kst),
                  src_func(i      , j - jst, k - kst),
                  src_func(i - ist, j - jst, k - kst));
              const Real abs_coeff_u = bilinear_interpolation(cc_x, cc_y,
                  abs_coeff(i      , j      , k - kst),
                  abs_coeff(i - ist, j      , k - kst),
                  abs_coeff(i      , j - jst, k - kst),
                  abs_coeff(i - ist, j - jst, k - kst));
              // clang-format on

              // Integrate along a ray segment
              const Real dtau = integrate_optical_thickness(
                  abs_coeff_u, abs_coeff(i, j, k), ds);
              rint(i_ray, i, j, k) = integrate_radiative_intensity(
                  rint_u, src_func_u, src_func(i, j, k), dtau);
            }
          }
        }
      }
    }
  }

  /// @brief Get maximum difference from previous state
  /// TODO: Reduce the number of evaluation (evaluate only at boundaries).
  Real get_max_diff(const Grid<Real> &grid) const {
    Real max_diff = 0.0;
    for (int i_ray = 0; i_ray < ang_quad.num_rays; ++i_ray) {
      for (int i = ib0; i <= ib1; ++i) {
        for (int j = jb0; j <= jb1; ++j) {
          for (int k = kb0; k <= kb1; ++k) {
            const Real diff =
                std::abs(rint(i_ray, i, j, k) - rint_old(i_ray, i, j, k));
            max_diff = util::max2(max_diff, diff);
          }
        }
      }
    }
    return max_diff;
  }

  /// @brief Solve radiative transfer equation
  /// @details All necessary information should be stored in the `rte_t` object.
  template <typename BoundaryCondition>
  void solve(const Grid<Real> &grid, const mpi::Shape &mpi_shape,
             const Real tolerance, const int max_iters, BoundaryCondition &&bc) {
    for (int iter = 0; iter < max_iters; ++iter) {
      rint_old.copy_from(rint);
      mpi_exchange_halo(grid, mpi_shape);
      std::forward<BoundaryCondition>(bc)(*this, grid, mpi_shape);
      single_sweep(grid);

      const Real max_diff = get_max_diff(grid);
      Real global_max_diff = 0.0;
      MPI_Allreduce(&max_diff, &global_max_diff, 1, mpi::data_type<Real>(),
                    MPI_MAX, mpi_shape.cart_comm);
      if (global_max_diff < tolerance)
        return;
    }

    if (mpi::is_root())
      std::printf("  RT did not converge in %d iterations.\n", max_iters);
  }

  /// @brief Integrate the optical thickness along a ray segment.
  constexpr Real integrate_optical_thickness(const Real abs_coeff_u,
                                             const Real abs_coeff_d,
                                             const Real ds) const noexcept {
    return 0.5 * (abs_coeff_u + abs_coeff_d) * ds;
  }

  /// @brief Integrate the radiative intensity along a ray segment.
  /// TODO: Switch eps_tau between single and double precision.
  inline Real integrate_radiative_intensity(const Real rint_u,
                                            const Real src_func_u,
                                            const Real src_func_d,
                                            const Real dtau) const noexcept {
    static constexpr Real eps_tau = 1.e-4;  // for double precision
    const Real exp_dtau = std::exp(-dtau);  // std::exp() is not constexpr.
    const Real u0 = 1.0 - exp_dtau;
    if (dtau < eps_tau) {
      const Real u1 = 0.5 * dtau;
      return rint_u * exp_dtau + (u0 - u1) * src_func_u + u1 * src_func_d;
    } else {
      const Real u1 = 1.0 - (u0 / dtau);
      return rint_u * exp_dtau + (u0 - u1) * src_func_u + u1 * src_func_d;
    }
  }

  /// @brief Bilinear interpolation
  constexpr Real bilinear_interpolation(const Real cx, const Real cy,
                                        const Real u00, const Real u10,
                                        const Real u01,
                                        const Real u11) const noexcept {
    // clang-format off
    return (1.0 - cx) * (1.0 - cy) * u00
         +         cx * (1.0 - cy) * u10
         + (1.0 - cx) *         cy * u01
         +         cx *         cy * u11;
    // clang-format on
  }
};

}  // namespace rt
}  // namespace miso
