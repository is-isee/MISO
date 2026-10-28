///
/// @brief Radiative heating rate from the RT solution (CPU)
/// @details Sets the source function and the absorption coefficient from a
/// table EOS, and computes the radiative heating rate Q_rad from the intensity
/// solved by `rt::RT`.
///
#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "array3d.hpp"
#include "constants.hpp"
#include "eos_table.hpp"
#include "grid.hpp"
#include "mpi_util.hpp"
#include "rt_host.hpp"
#include "types.hpp"
#include "utility.hpp"

namespace miso {
namespace rt {

/// @brief Stefan-Boltzmann constant [erg cm^-2 s^-1 K^-4]
template <typename Real>
inline constexpr Real stefan_boltzmann = static_cast<Real>(5.670374419e-5);

/// @brief Set S = sigma T^4 / pi and alpha = rho kappa in all cells.
/// @param ro density [g/cm^3]
/// @param ei specific internal energy [erg/g]
/// @details Throws if alpha or S is not finite, e.g. where the opacity table
/// has `inf` (see `eos::TableEOS`).
template <typename Real, typename Backend>
void set_source_and_absorption(RT<Real> &rt, const Grid<Real> &grid,
                               const Array3D<Real> &ro, const Array3D<Real> &ei,
                               const eos::TableEOS<Real, Backend> &eos) {
  for (int i = 0; i < grid.i_total; ++i) {
    for (int j = 0; j < grid.j_total; ++j) {
      for (int k = 0; k < grid.k_total; ++k) {
        const Real te = eos.temperature(ro(i, j, k), ei(i, j, k));
        rt.src_func(i, j, k) =
            stefan_boltzmann<Real> * util::pow2(util::pow2(te)) / pi<Real>;
        rt.abs_coeff(i, j, k) =
            ro(i, j, k) * eos.opacity(ro(i, j, k), ei(i, j, k));
        if (!std::isfinite(rt.abs_coeff(i, j, k)) ||
            !std::isfinite(rt.src_func(i, j, k))) {
          throw std::runtime_error(
              "Opacity or temperature from the table is not finite at ro = " +
              std::to_string(ro(i, j, k)) +
              ", ei = " + std::to_string(ei(i, j, k)));
        }
      }
    }
  }
}

///
/// @brief Radiative heating rate Q_rad [erg cm^-3 s^-1]
/// @details Q_rad is computed in two ways and blended by the vertical optical
/// depth tau, as in Hotta & Iijima (2020, Appendix C):
///
///  - Q_J = 4 pi alpha (J - S), accurate where the medium is optically thin.
///  - Q_F = -div F, accurate where the medium is optically thick, because
///    J - S is a small difference of large numbers there.
///  - Q_rad = (1 - w) Q_J + w Q_F, where w goes from 0 at tau = tau_thin to 1
///    at tau = tau_thick, linearly in ln tau.
///  - Q_rad = 0 where tau < tau_gate.
///
/// The z axis is the vertical direction and points upward. tau is integrated
/// from the top of the domain (tau = 0 at the topmost cell center), so the z
/// direction must not be periodic.
///
/// Values are defined at the RT points (cell centers from `rt.kb0` to
/// `rt.kb1`, etc.). Q_rad is set in the physical cells only.
///
template <typename Real> struct RadiativeHeating {
  /// @brief Vertical optical depth from the top of the domain
  Array3D<Real> tau;
  /// @brief Mean intensity J
  Array3D<Real> mean_intensity;
  /// @brief Radiative flux F
  Array3D<Real> flux_x, flux_y, flux_z;
  /// @brief Radiative heating rate Q_rad
  Array3D<Real> qrad;

  /// @brief Optical depths for blending Q_J and Q_F
  Real tau_thin = Real(1.e-2), tau_thick = Real(2.e-1);
  /// @brief Q_rad is set to zero where tau < tau_gate
  Real tau_gate = Real(1.e-4);

  explicit RadiativeHeating(const Grid<Real> &grid)
      : tau(grid.i_total, grid.j_total, grid.k_total),
        mean_intensity(grid.i_total, grid.j_total, grid.k_total),
        flux_x(grid.i_total, grid.j_total, grid.k_total),
        flux_y(grid.i_total, grid.j_total, grid.k_total),
        flux_z(grid.i_total, grid.j_total, grid.k_total),
        qrad(grid.i_total, grid.j_total, grid.k_total) {
    util::clear_array(tau);
    util::clear_array(mean_intensity);
    util::clear_array(flux_x);
    util::clear_array(flux_y);
    util::clear_array(flux_z);
    util::clear_array(qrad);
  }

  /// @brief Compute Q_rad from the intensity solved by `rt.solve`.
  void compute(const RT<Real> &rt, const Grid<Real> &grid,
               const mpi::Shape &mpi_shape) {
    if (grid.k_size == 1) {
      throw std::runtime_error("RadiativeHeating needs the z direction");
    }
    if (rt.ang_quad.num_rays != 24) {
      // Otherwise the quadrature is a single ray in +z, which gives no J or F.
      throw std::runtime_error("RadiativeHeating needs 24 rays");
    }
    compute_optical_depth(rt, grid, mpi_shape);
    compute_moments(rt);
    if (grid.i_size > 1) {
      fill_upper_plane(flux_x, grid, mpi_shape, Direction::X);
    }
    if (grid.j_size > 1) {
      fill_upper_plane(flux_y, grid, mpi_shape, Direction::Y);
    }
    fill_upper_plane(flux_z, grid, mpi_shape, Direction::Z);

    const bool x_upper = mpi_shape.x_procs_pos != MPI_PROC_NULL;
    const bool y_upper = mpi_shape.y_procs_pos != MPI_PROC_NULL;
    const bool z_upper = mpi_shape.z_procs_pos != MPI_PROC_NULL;
    const Real log_tau_thin = std::log(tau_thin);
    const Real d_log_tau = std::log(tau_thick) - log_tau_thin;

    util::clear_array(qrad);
    for (int i = rt.ib0 + grid.is; i <= rt.ib1; ++i) {
      for (int j = rt.jb0 + grid.js; j <= rt.jb1; ++j) {
        for (int k = rt.kb0 + grid.ks; k <= rt.kb1; ++k) {
          Real div_f = derivative(flux_z, grid.z, i, j, k, 0, 0, 1,
                                  k == rt.kb1 && !z_upper);
          if (grid.i_size > 1) {
            div_f += derivative(flux_x, grid.x, i, j, k, 1, 0, 0,
                                i == rt.ib1 && !x_upper);
          }
          if (grid.j_size > 1) {
            div_f += derivative(flux_y, grid.y, i, j, k, 0, 1, 0,
                                j == rt.jb1 && !y_upper);
          }
          const Real q_f = -div_f;
          const Real q_j = Real(4) * pi<Real> * rt.abs_coeff(i, j, k) *
                           (mean_intensity(i, j, k) - rt.src_func(i, j, k));

          const Real t = tau(i, j, k);
          const Real w = util::clamp<Real>(
              (std::log(t) - log_tau_thin) / d_log_tau, Real(0), Real(1));
          qrad(i, j, k) =
              (t < tau_gate) ? Real(0) : (Real(1) - w) * q_j + w * q_f;
        }
      }
    }
  }

private:
  /// @brief Integrate alpha downward from the top of the domain.
  /// @details The ranks are processed from the top: each rank receives tau at
  /// its topmost point from the rank above, and sends tau at its lowest point
  /// (which is the topmost point of the rank below) to the rank below.
  void compute_optical_depth(const RT<Real> &rt, const Grid<Real> &grid,
                             const mpi::Shape &mpi_shape) {
    int dims[3], periods[3], coords[3];
    MPI_Cart_get(mpi_shape.cart_comm, 3, dims, periods, coords);
    if (periods[2]) {
      throw std::runtime_error("RadiativeHeating needs a non-periodic z "
                               "direction");
    }

    std::vector<Real> plane(grid.i_total * grid.j_total, Real(0));
    if (mpi_shape.z_procs_pos != MPI_PROC_NULL) {
      MPI_Recv(plane.data(), static_cast<int>(plane.size()),
               mpi::data_type<Real>(), mpi_shape.z_procs_pos, 2100,
               mpi_shape.cart_comm, MPI_STATUS_IGNORE);
    }
    for (int i = 0; i < grid.i_total; ++i) {
      for (int j = 0; j < grid.j_total; ++j) {
        tau(i, j, rt.kb1) = plane[i * grid.j_total + j];
        for (int k = rt.kb1 - 1; k >= rt.kb0; --k) {
          tau(i, j, k) = tau(i, j, k + 1) +
                         Real(0.5) *
                             (rt.abs_coeff(i, j, k) + rt.abs_coeff(i, j, k + 1)) *
                             (grid.z[k + 1] - grid.z[k]);
        }
        plane[i * grid.j_total + j] = tau(i, j, rt.kb0);
      }
    }
    if (mpi_shape.z_procs_neg != MPI_PROC_NULL) {
      MPI_Send(plane.data(), static_cast<int>(plane.size()),
               mpi::data_type<Real>(), mpi_shape.z_procs_neg, 2100,
               mpi_shape.cart_comm);
    }
  }

  /// @brief J = sum_m w_m I_m and F = 4 pi sum_m w_m mu_m I_m (sum of w_m = 1)
  void compute_moments(const RT<Real> &rt) {
    const auto &q = rt.ang_quad;
    for (int i = rt.ib0; i <= rt.ib1; ++i) {
      for (int j = rt.jb0; j <= rt.jb1; ++j) {
        for (int k = rt.kb0; k <= rt.kb1; ++k) {
          Real jj = 0, fx = 0, fy = 0, fz = 0;
          for (int m = 0; m < q.num_rays; ++m) {
            const Real wi = q.weights[m] * rt.rint(m, i, j, k);
            jj += wi;
            fx += q.mu_x[m] * wi;
            fy += q.mu_y[m] * wi;
            fz += q.mu_z[m] * wi;
          }
          mean_intensity(i, j, k) = jj;
          flux_x(i, j, k) = Real(4) * pi<Real> * fx;
          flux_y(i, j, k) = Real(4) * pi<Real> * fy;
          flux_z(i, j, k) = Real(4) * pi<Real> * fz;
        }
      }
    }
  }

  /// @brief Fill the plane next to the last RT point (index b1 + 1) with the
  /// first physical plane of the neighbor on the positive side.
  void fill_upper_plane(Array3D<Real> &arr, const Grid<Real> &grid,
                        const mpi::Shape &mpi_shape, Direction direction) {
    const int n0 = grid.i_total, n1 = grid.j_total, n2 = grid.k_total;
    int send_index = 0, recv_index = 0, pos = 0, neg = 0, size = 0;
    switch (direction) {
    case Direction::X:
      send_index = grid.i_margin;
      recv_index = grid.i_margin + grid.i_size;
      pos = mpi_shape.x_procs_pos;
      neg = mpi_shape.x_procs_neg;
      size = n1 * n2;
      break;
    case Direction::Y:
      send_index = grid.j_margin;
      recv_index = grid.j_margin + grid.j_size;
      pos = mpi_shape.y_procs_pos;
      neg = mpi_shape.y_procs_neg;
      size = n0 * n2;
      break;
    case Direction::Z:
      send_index = grid.k_margin;
      recv_index = grid.k_margin + grid.k_size;
      pos = mpi_shape.z_procs_pos;
      neg = mpi_shape.z_procs_neg;
      size = n0 * n1;
      break;
    }

    auto element = [&](int a, int b, int index) -> Real & {
      switch (direction) {
      case Direction::X:
        return arr(index, a, b);
      case Direction::Y:
        return arr(a, index, b);
      default:
        return arr(a, b, index);
      }
    };
    const int na = (direction == Direction::X) ? n1 : n0;
    const int nb = (direction == Direction::Z) ? n1 : n2;

    std::vector<Real> send(size), recv(size);
    for (int a = 0; a < na; ++a) {
      for (int b = 0; b < nb; ++b) {
        send[a * nb + b] = element(a, b, send_index);
      }
    }
    const int tag = 2200 + static_cast<int>(direction);
    MPI_Sendrecv(send.data(), size, mpi::data_type<Real>(), neg, tag, recv.data(),
                 size, mpi::data_type<Real>(), pos, tag, mpi_shape.cart_comm,
                 MPI_STATUS_IGNORE);
    if (pos != MPI_PROC_NULL) {
      for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
          element(a, b, recv_index) = recv[a * nb + b];
        }
      }
    }
  }

  /// @brief Derivative of f along (di, dj, dk) at (i, j, k). Central
  /// difference, or backward difference at the upper physical boundary.
  static Real derivative(const Array3D<Real> &f, const std::vector<Real> &s,
                         int i, int j, int k, int di, int dj, int dk,
                         bool backward) {
    const int n = i * di + j * dj + k * dk;
    if (backward) {
      return (f(i, j, k) - f(i - di, j - dj, k - dk)) / (s[n] - s[n - 1]);
    }
    return (f(i + di, j + dj, k + dk) - f(i - di, j - dj, k - dk)) /
           (s[n + 1] - s[n - 1]);
  }
};

}  // namespace rt
}  // namespace miso
