#pragma once
#include <miso/mhd_model_base.hpp>
#include <stdexcept>

using namespace miso;

template <typename Real> struct InitialCondition {
  eos::IdealEOS<Real> &eos;
  Real rol, prl, vvl;
  Real ror, prr, vvr;

  explicit InitialCondition(Config &config, eos::IdealEOS<Real> &eos)
      : eos(eos), rol(config["shock_tube"]["rol"].as<Real>()),
        prl(config["shock_tube"]["prl"].as<Real>()),
        vvl(config["shock_tube"]["vvl"].as<Real>()),
        ror(config["shock_tube"]["ror"].as<Real>()),
        prr(config["shock_tube"]["prr"].as<Real>()),
        vvr(config["shock_tube"]["vvr"].as<Real>()) {}

  // The signature must not be changed as it is called inside miso::mhd::MHD.
  void apply(mhd::FieldsView<Real> qq, GridView<const Real> grid) const {

    for (int k = 0; k < grid.k_total; ++k) {
      for (int j = 0; j < grid.j_total; ++j) {
        for (int i = 0; i < grid.i_total; ++i) {
          qq.vx(i, j, k) = 0.0;
          qq.vy(i, j, k) = 0.0;
          qq.vz(i, j, k) = 0.0;
          qq.bx(i, j, k) = 0.0;
          qq.by(i, j, k) = 0.0;
          qq.bz(i, j, k) = 0.0;
          qq.ph(i, j, k) = 0.0;

          Real xyz;
          if (grid.i_total > 1) {
            xyz = grid.x[i];
          } else if (grid.j_total > 1) {
            xyz = grid.y[j];
          } else if (grid.k_total > 1) {
            xyz = grid.z[k];
          } else {
            throw std::runtime_error(
                "At least one of grid dimensions must be greater than 1.");
          }

          if (xyz < 0.5) {
            qq.ro(i, j, k) = rol;
            qq.ei(i, j, k) = prl / (eos.gm - 1.0) / qq.ro(i, j, k);
            qq.vx(i, j, k) = vvl;
          } else {
            qq.ro(i, j, k) = ror;
            qq.ei(i, j, k) = prr / (eos.gm - 1.0) / qq.ro(i, j, k);
            qq.vx(i, j, k) = vvr;
          }
        }
      }
    }
  }
};
