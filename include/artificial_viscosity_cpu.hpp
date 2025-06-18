#pragma once

#include <algorithm>
#include <initializer_list>
#include <cassert>
#include <string>

#include "model.hpp"
#include "constants.hpp"
#include "artificial_viscosity_core.hpp"

template <typename Real>
struct ArtificialViscosity {
    Config& config;
    Time<Real>& time;
    Grid<Real>& grid;
    EOS<Real>& eos;
    MHD<Real>& mhd;

    Array3D<Real> cc;
    Real ep, fh;
    Real cs_fac, ca_fac, vv_fac;

    ArtificialViscosity(Model<Real>& model)
        : config(model.config),
          time(model.time),
          grid(model.grid_local),
          eos(model.eos),
          mhd(model.mhd),
          cc(grid.i_total, grid.j_total, grid.k_total) {
        this->ep = config.yaml_obj["artificial_viscosity"]["ep"].template as<Real>();
        this->fh = config.yaml_obj["artificial_viscosity"]["fh"].template as<Real>();
        this->cs_fac = config.yaml_obj["artificial_viscosity"]["cs_fac"].template as<Real>();
        this->ca_fac = config.yaml_obj["artificial_viscosity"]["ca_fac"].template as<Real>();
        this->vv_fac = config.yaml_obj["artificial_viscosity"]["vv_fac"].template as<Real>();
        assert(ep >= 0);
        assert(fh >= 0);
    }

    void characteristic_velocity_eval() {
        for(int i = 0; i < grid.i_total; ++i) {
            for(int j = 0; j < grid.j_total; ++j) {
                for(int k = 0; k < grid.k_total; ++k) {
                    // cs: sound speed, vv: fluid velocity, ca: Alfvén speed
                    Real cs = std::sqrt(this->eos.gm*(this->eos.gm-1.0)*this->mhd.qq.ei(i,j,k));
                    Real vv = std::sqrt(
                        + this->mhd.qq.vx(i,j,k)*this->mhd.qq.vx(i,j,k)
                        + this->mhd.qq.vy(i,j,k)*this->mhd.qq.vy(i,j,k)
                        + this->mhd.qq.vz(i,j,k)*this->mhd.qq.vz(i,j,k)
                    );  
                    Real ca = std::sqrt( (
                        + this->mhd.qq.bx(i,j,k)*this->mhd.qq.bx(i,j,k)
                        + this->mhd.qq.by(i,j,k)*this->mhd.qq.by(i,j,k)
                        + this->mhd.qq.bz(i,j,k)*this->mhd.qq.bz(i,j,k)
                    )/this->mhd.qq.ro(i,j,k)*pii4<Real>);
                    this->cc(i,j,k) = cs*this->cs_fac + vv*this->vv_fac + ca*this->ca_fac;
                }
            }
        }
    }

    void update(MHDCore<Real>& qq, MHDCore<Real>& qq_rslt, Array3D<Real>& cc, std::vector<Real>& dxyzi, std::string direction) {
        int i0_ = 0; int i1_ = grid.i_total; int is = 0;
        int j0_ = 0; int j1_ = grid.j_total; int js = 0;
        int k0_ = 0; int k1_ = grid.k_total; int ks = 0;

        if(direction == "x") {
            i0_ = 2*grid.is; i1_ = grid.i_total - 2*grid.is; is = grid.is;
        } else if(direction == "y") {
            j0_ = 2*grid.js; j1_ = grid.j_total - 2*grid.js; js = grid.js;
        } else if(direction == "z") {
            k0_ = 2*grid.ks; k1_ = grid.k_total - 2*grid.ks; ks = grid.ks;
        } 

        Real qql2, qql1, qqc, qqr1, qqr2;
        Real ccl, ccc, ccr;
        Real dqq_dw, dqq_cn, dqq_up;

        for(int i = i0_; i < i1_; ++i) {
            for(int j = j0_; j < j1_; ++j) {
                for(int k = k0_; k < k1_; ++k) {
                    // chracteristic velocity
                    ccl  = cc(i-  is, j-  js, k-  ks);
                    ccc  = cc(i     , j     , k     );
                    ccr  = cc(i+  is, j+  js, k+  ks);

                    // density
                    qql2 = qq.ro(i-2*is, j-2*js, k-2*ks);
                    qql1 = qq.ro(i-  is, j-  js, k-  ks);
                    qqc  = qq.ro(i     , j     , k     );
                    qqr1 = qq.ro(i+  is, j+  js, k+  ks);
                    qqr2 = qq.ro(i+2*is, j+2*js, k+2*ks);
                    // dqq at i-is, j-js, k-2ks
                    dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, this->ep);
                    // dqq at i, j, k
                    dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc , qqr1, this->ep);
                    // dqq at i+is, j+js, k+ks
                    dqq_up = artificial_viscosity::dqq_eval(qqc , qqr1, qqr2, this->ep);
                    Real fro_dw = artificial_viscosity::flux_core(qql1, qqc , dqq_dw, dqq_cn, Real(0.5)*(ccl + ccc), fh);
                    Real fro_up = artificial_viscosity::flux_core(qqc , qqr1, dqq_cn, dqq_up, Real(0.5)*(ccc + ccr), fh);

                    qq_rslt.ro(i, j, k) = qq.ro(i, j, k) - (fro_up - fro_dw)*dxyzi[i*is + j*js + k*ks]*time.dt;

                    // x momentum
                    qql2 = qq.ro(i-2*is, j-2*js, k-2*ks)*qq.vx(i-2*is, j-2*js, k-2*ks);
                    qql1 = qq.ro(i-  is, j-  js, k-  ks)*qq.vx(i-  is, j-  js, k-  ks);
                    qqc  = qq.ro(i     , j     , k     )*qq.vx(i     , j     , k     );
                    qqr1 = qq.ro(i+  is, j+  js, k+  ks)*qq.vx(i+  is, j+  js, k+  ks);
                    qqr2 = qq.ro(i+2*is, j+2*js, k+2*ks)*qq.vx(i+2*is, j+2*js, k+2*ks);
                    // dqq at i-is, j-js, k-2ks
                    dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, this->ep);
                    // dqq at i, j, k
                    dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc , qqr1, this->ep);
                    // dqq at i+is, j+js, k+ks
                    dqq_up = artificial_viscosity::dqq_eval(qqc , qqr1, qqr2, this->ep);
                    Real frx_dw = artificial_viscosity::flux_core(qql1, qqc , dqq_dw, dqq_cn, Real(0.5)*(ccl + ccc), fh);
                    Real frx_up = artificial_viscosity::flux_core(qqc , qqr1, dqq_cn, dqq_up, Real(0.5)*(ccc + ccr), fh);

                    qq_rslt.vx(i, j, k) = (
                        qq.ro(i, j, k)*qq.vx(i,j,k) - (frx_up - frx_dw)*dxyzi[i*is + j*js + k*ks]*time.dt
                    )/qq_rslt.ro(i, j, k);

                    // y momentum
                    qql2 = qq.ro(i-2*is, j-2*js, k-2*ks)*qq.vy(i-2*is, j-2*js, k-2*ks);
                    qql1 = qq.ro(i-  is, j-  js, k-  ks)*qq.vy(i-  is, j-  js, k-  ks);
                    qqc  = qq.ro(i     , j     , k     )*qq.vy(i     , j     , k     );
                    qqr1 = qq.ro(i+  is, j+  js, k+  ks)*qq.vy(i+  is, j+  js, k+  ks);
                    qqr2 = qq.ro(i+2*is, j+2*js, k+2*ks)*qq.vy(i+2*is, j+2*js, k+2*ks);
                    // dqq at i-is, j-js, k-2ks
                    dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, this->ep);
                    // dqq at i, j, k
                    dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc , qqr1, this->ep);
                    // dqq at i+is, j+js, k+ks
                    dqq_up = artificial_viscosity::dqq_eval(qqc , qqr1, qqr2, this->ep);
                    Real fry_dw = artificial_viscosity::flux_core(qql1, qqc , dqq_dw, dqq_cn, Real(0.5)*(ccl + ccc), fh);
                    Real fry_up = artificial_viscosity::flux_core(qqc , qqr1, dqq_cn, dqq_up, Real(0.5)*(ccc + ccr), fh);

                    qq_rslt.vy(i, j, k) = (
                        qq.ro(i, j, k)*qq.vy(i,j,k) - (fry_up - fry_dw)*dxyzi[i*is + j*js + k*ks]*time.dt
                    )/qq_rslt.ro(i, j, k);

                    // z momentum
                    qql2 = qq.ro(i-2*is, j-2*js, k-2*ks)*qq.vz(i-2*is, j-2*js, k-2*ks);
                    qql1 = qq.ro(i-  is, j-  js, k-  ks)*qq.vz(i-  is, j-  js, k-  ks);
                    qqc  = qq.ro(i     , j     , k     )*qq.vz(i     , j     , k     );
                    qqr1 = qq.ro(i+  is, j+  js, k+  ks)*qq.vz(i+  is, j+  js, k+  ks);
                    qqr2 = qq.ro(i+2*is, j+2*js, k+2*ks)*qq.vz(i+2*is, j+2*js, k+2*ks);
                    // dqq at i-is, j-js, k-2ks
                    dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, this->ep);
                    // dqq at i, j, k
                    dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc , qqr1, this->ep);
                    // dqq at i+is, j+js, k+ks
                    dqq_up = artificial_viscosity::dqq_eval(qqc , qqr1, qqr2, this->ep);
                    Real frz_dw = artificial_viscosity::flux_core(qql1, qqc , dqq_dw, dqq_cn, Real(0.5)*(ccl + ccc), fh);
                    Real frz_up = artificial_viscosity::flux_core(qqc , qqr1, dqq_cn, dqq_up, Real(0.5)*(ccc + ccr), fh);

                    qq_rslt.vz(i, j, k) = (
                        qq.ro(i, j, k)*qq.vz(i,j,k) - (frz_up - frz_dw)*dxyzi[i*is + j*js + k*ks]*time.dt
                    )/qq_rslt.ro(i, j, k);

                    // x magnetic field
                    qql2 = qq.bx(i-2*is, j-2*js, k-2*ks);
                    qql1 = qq.bx(i-  is, j-  js, k-  ks);
                    qqc  = qq.bx(i     , j     , k     );
                    qqr1 = qq.bx(i+  is, j+  js, k+  ks);
                    qqr2 = qq.bx(i+2*is, j+2*js, k+2*ks);
                    // dqq at i-is, j-js, k-2ks
                    dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, this->ep);
                    // dqq at i, j, k
                    dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc , qqr1, this->ep);
                    // dqq at i+is, j+js, k+ks
                    dqq_up = artificial_viscosity::dqq_eval(qqc , qqr1, qqr2, this->ep);
                    Real fbx_dw = artificial_viscosity::flux_core(qql1, qqc , dqq_dw, dqq_cn, Real(0.5)*(ccl + ccc), fh);
                    Real fbx_up = artificial_viscosity::flux_core(qqc , qqr1, dqq_cn, dqq_up, Real(0.5)*(ccc + ccr), fh);

                    qq_rslt.bx(i, j, k) = qq.bx(i, j, k) - (fbx_up - fbx_dw)*dxyzi[i*is + j*js + k*ks]*time.dt;

                    // y magnetic field
                    qql2 = qq.by(i-2*is, j-2*js, k-2*ks);
                    qql1 = qq.by(i-  is, j-  js, k-  ks);
                    qqc  = qq.by(i     , j     , k     );
                    qqr1 = qq.by(i+  is, j+  js, k+  ks);
                    qqr2 = qq.by(i+2*is, j+2*js, k+2*ks);
                    // dqq at i-is, j-js, k-2ks
                    dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, this->ep);
                    // dqq at i, j, k
                    dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc , qqr1, this->ep);
                    // dqq at i+is, j+js, k+ks
                    dqq_up = artificial_viscosity::dqq_eval(qqc , qqr1, qqr2, this->ep);
                    Real fby_dw = artificial_viscosity::flux_core(qql1, qqc , dqq_dw, dqq_cn, Real(0.5)*(ccl + ccc), fh);
                    Real fby_up = artificial_viscosity::flux_core(qqc , qqr1, dqq_cn, dqq_up, Real(0.5)*(ccc + ccr), fh);

                    qq_rslt.by(i, j, k) = qq.by(i, j, k) - (fby_up - fby_dw)*dxyzi[i*is + j*js + k*ks]*time.dt;

                    // z magnetic field
                    qql2 = qq.bz(i-2*is, j-2*js, k-2*ks);
                    qql1 = qq.bz(i-  is, j-  js, k-  ks);
                    qqc  = qq.bz(i     , j     , k     );
                    qqr1 = qq.bz(i+  is, j+  js, k+  ks);
                    qqr2 = qq.bz(i+2*is, j+2*js, k+2*ks);
                    // dqq at i-is, j-js, k-2ks
                    dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, this->ep);
                    // dqq at i, j, k
                    dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc , qqr1, this->ep);
                    // dqq at i+is, j+js, k+ks
                    dqq_up = artificial_viscosity::dqq_eval(qqc , qqr1, qqr2, this->ep);
                    Real fbz_dw = artificial_viscosity::flux_core(qql1, qqc , dqq_dw, dqq_cn, Real(0.5)*(ccl + ccc), fh);
                    Real fbz_up = artificial_viscosity::flux_core(qqc , qqr1, dqq_cn, dqq_up, Real(0.5)*(ccc + ccr), fh);

                    qq_rslt.bz(i, j, k) = qq.bz(i, j, k) - (fbz_up - fbz_dw)*dxyzi[i*is + j*js + k*ks]*time.dt;

                    // z magnetic field
                    qql2 = qq.ph(i-2*is, j-2*js, k-2*ks);
                    qql1 = qq.ph(i-  is, j-  js, k-  ks);
                    qqc  = qq.ph(i     , j     , k     );
                    qqr1 = qq.ph(i+  is, j+  js, k+  ks);
                    qqr2 = qq.ph(i+2*is, j+2*js, k+2*ks);
                    // dqq at i-is, j-js, k-2ks
                    dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, this->ep);
                    // dqq at i, j, k
                    dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc , qqr1, this->ep);
                    // dqq at i+is, j+js, k+ks
                    dqq_up = artificial_viscosity::dqq_eval(qqc , qqr1, qqr2, this->ep);
                    Real fph_dw = artificial_viscosity::flux_core(qql1, qqc , dqq_dw, dqq_cn, Real(0.5)*(ccl + ccc), fh);
                    Real fph_up = artificial_viscosity::flux_core(qqc , qqr1, dqq_cn, dqq_up, Real(0.5)*(ccc + ccr), fh);

                    qq_rslt.ph(i, j, k) = qq.ph(i, j, k) - (fph_up - fph_dw)*dxyzi[i*is + j*js + k*ks]*time.dt;

                    // total energy
                    qql2 = qq.ro(i-2*is, j-2*js, k-2*ks)*qq.ei(i-2*is, j-2*js, k-2*ks);
                    qql1 = qq.ro(i-  is, j-  js, k-  ks)*qq.ei(i-  is, j-  js, k-  ks);
                    qqc  = qq.ro(i     , j     , k     )*qq.ei(i     , j     , k     );
                    qqr1 = qq.ro(i+  is, j+  js, k+  ks)*qq.ei(i+  is, j+  js, k+  ks);
                    qqr2 = qq.ro(i+2*is, j+2*js, k+2*ks)*qq.ei(i+2*is, j+2*js, k+2*ks);
                    // dqq at i-is, j-js, k-2ks
                    dqq_dw = artificial_viscosity::dqq_eval(qql2, qql1, qqc, this->ep);
                    // dqq at i, j, k
                    dqq_cn = artificial_viscosity::dqq_eval(qql1, qqc , qqr1, this->ep);
                    // dqq at i+is, j+js, k+ks
                    dqq_up = artificial_viscosity::dqq_eval(qqc , qqr1, qqr2, this->ep);
                    Real fei_dw = artificial_viscosity::flux_core(qql1, qqc , dqq_dw, dqq_cn, Real(0.5)*(ccl + ccc), fh);
                    Real fei_up = artificial_viscosity::flux_core(qqc , qqr1, dqq_cn, dqq_up, Real(0.5)*(ccc + ccr), fh);

                    // Et: Total energy per unit volume, note that ei is internal energy per unit mass
                    Real Et = qq.ro(i, j, k)*qq.ei(i, j, k)
                        + 0.5*qq.ro(i, j, k)*(
                              qq.vx(i, j, k)*qq.vx(i, j, k)
                            + qq.vy(i, j, k)*qq.vy(i, j, k)
                            + qq.vz(i, j, k)*qq.vz(i, j, k)
                        )
                        + pii8<Real>*(
                              qq.bx(i, j, k)*qq.bx(i, j, k)
                            + qq.by(i, j, k)*qq.by(i, j, k)
                            + qq.bz(i, j, k)*qq.bz(i, j, k)
                        );

                    qq_rslt.ei(i, j, k) = (
                        Et - (fei_up - fei_dw)*dxyzi[i*is + j*js + k*ks]*time.dt
                        - 0.5*qq_rslt.ro(i, j, k)*(
                              qq_rslt.vx(i, j, k)*qq_rslt.vx(i, j, k)
                            + qq_rslt.vy(i, j, k)*qq_rslt.vy(i, j, k)
                            + qq_rslt.vz(i, j, k)*qq_rslt.vz(i, j, k) )
                        - pii8<Real>*(
                              qq_rslt.bx(i, j, k)*qq_rslt.bx(i, j, k)
                            + qq_rslt.by(i, j, k)*qq_rslt.by(i, j, k)
                            + qq_rslt.bz(i, j, k)*qq_rslt.bz(i, j, k) )
                    )/qq_rslt.ro(i, j, k);
                }
            }
        }
    }    
};