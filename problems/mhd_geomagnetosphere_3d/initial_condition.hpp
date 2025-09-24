#pragma once

#include "constants.hpp"
#include "model.hpp"
#include <force.hpp>

template <typename Real> struct InitialCondition {
  Config &config;
  Grid<Real> &grid;
  EOS<Real> &eos;

  InitialCondition(Model<Real> &model)
      : config(model.config), grid(model.grid_local), eos(model.eos) {}
  void apply(MHDCore<Real> &qq) {

    for (int i = 0; i < grid.i_total; ++i) {
      for (int j = 0; j < grid.j_total; ++j) {
        for (int k = 0; k < grid.k_total; ++k) {

          // clang-format off
          Real ro_sw = config.yaml_obj["solar_wind"]["ro_sw"].template as<Real>();
          Real pr_sw = config.yaml_obj["solar_wind"]["pr_sw"].template as<Real>();
          Real vx_sw = config.yaml_obj["solar_wind"]["vx_sw"].template as<Real>();
          Real bz_imf = config.yaml_obj["solar_wind"]["bz_imf"].template as<Real>();
          // clang-format on
          // distance from the earth center
          Real rr = std::sqrt(grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] +
                              grid.z[k] * grid.z[k]);

          // density
          Real ro_tmp = util::pow3(1 / rr);
          Real ro_sw_thresh = 0.2 * ro_sw;
          if (ro_tmp > ro_sw_thresh) {
            qq.ro(i, j, k) = ro_tmp;
          } else {
            qq.ro(i, j, k) = ro_sw_thresh;
          }

          // dipole magnetic field
          Real rro5 = util::pow5(1 / rr);
          qq.bx(i, j, k) = -rro5 * 3.0 * grid.x[i] * grid.z[k];
          qq.by(i, j, k) = -rro5 * 3.0 * grid.y[j] * grid.z[k];
          qq.bz(i, j, k) =
              +rro5 * (grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] -
                       2.0 * grid.z[k] * grid.z[k]);

          // pressure
          Real pr;
          Real pr_earth =
              config.yaml_obj["geo_boundary"]["pr_earth"].template as<Real>();
          Real pr_tmp = pr_earth * util::pow2(1 / rr);

          if (pr_tmp > pr_sw) {
            pr = pr_tmp;
          } else {
            pr = pr_sw;
          }
          qq.ei(i, j, k) = pr / (eos.gm - 1.0) / qq.ro(i, j, k);

          qq.vx(i, j, k) = 0.0;
          qq.vy(i, j, k) = 0.0;
          qq.vz(i, j, k) = 0.0;

          qq.ph(i, j, k) = 0.0;
        }
      }
    }
  }
};
