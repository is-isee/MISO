#include "common.hpp"

struct InitialCondition {
  Real ro_sw;
  Real pr_sw;
  Real vx_sw;
  Real bz_imf;
  Real pr_earth;

  explicit InitialCondition(Config &config) {
    ro_sw = config.yaml_obj["solar_wind"]["mass_density"].as<Real>();
    pr_sw = config.yaml_obj["solar_wind"]["gas_pressure"].as<Real>();
    vx_sw = config.yaml_obj["solar_wind"]["x_velocity_field"].as<Real>();
    bz_imf = config.yaml_obj["solar_wind"]["z_magnetic_field"].as<Real>();
    pr_earth = config.yaml_obj["magnetosphere"]["gas_pressure"].as<Real>();
  }

  // The signature must not be changed as it is called inside miso::mhd::MHD.
  template <typename EOS>
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid,
             const EOS &eos) const {
    for (int k = 0; k < grid.k_total; ++k) {
      for (int j = 0; j < grid.j_total; ++j) {
        for (int i = 0; i < grid.i_total; ++i) {
          // distance from the earth center
          Real rr = util::sqrt(grid.x[i] * grid.x[i] + grid.y[j] * grid.y[j] +
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
