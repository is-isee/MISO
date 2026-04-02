#pragma once
#include <cmath>
#include <cstddef>
#include <vector>

// This file provides the exact solution of Sod's shock tube problem, which is used for testing the correctness of the numerical solution in test_hd1d_shock_tube.cpp.

template <typename Real> class SodSolution {
private:
  Real pfunc(Real P, Real gm, Real csl, Real csr, Real prr, Real prl, Real vxl,
             Real vxr) {
    Real alpha = 2.0 * gm / (gm - 1);
    return std::sqrt(2.0 / gm) * (P - 1.0) /
               std::sqrt(gm - 1.0 + (gm + 1.0) * P) -
           2.0 / (gm - 1.0) * csl / csr *
               (1.0 - std::pow(prr / prl * P, 1.0 / alpha)) +
           (vxl - vxr) / csr;
  }

  Real get_P(Real gm, Real csl, Real csr, Real prr, Real prl, Real vxl,
             Real vxr) {
    Real p0 = 0.0;
    Real p2 = 10.0;
    Real p1 = (p0 + p2) * 0.5;
    Real a1;
    while (std::abs(p1 - p0) > 1.e-6) {
      a1 = pfunc(p1, gm, csl, csr, prr, prl, vxl, vxr);
      if (a1 > 0) {
        p2 = p1;
      } else {
        p0 = p1;
      }
      p1 = (p0 + p2) * 0.5;
    }
    return p1;
  }

public:
  std::vector<Real> x, ro, vx, pr;

  SodSolution(std::vector<Real> x_)
      : x(x_), ro(x_.size()), vx(x_.size()), pr(x_.size()) {}

  void calc_sod_solution(Real time, Real gm, Real xm, Real csl, Real csr,
                         Real ror, Real rol, Real prr, Real prl, Real vxl,
                         Real vxr) {
    Real P = get_P(gm, csl, csr, prr, prl, vxl, vxr);

    Real alpha = 2.0 * gm / (gm - 1.0);

    Real ro1 = ror;
    Real pr1 = prr;
    Real vx1 = vxr;

    Real pr2 = prr * P;
    Real vx2 = vxr + csr * std::sqrt(2.0 / gm) * (P - 1.0) /
                         std::sqrt(gm - 1.0 + (gm + 1.0) * P);
    Real ro2 = ror * (gm - 1.0 + (gm + 1.0) * P) / (gm + 1.0 + (gm - 1.0) * P);

    Real vs = vxr + (P - 1.0) * csr * csr / (gm * (vx2 - vxr));
    Real vc = vx2;

    Real pr3 = pr2;
    Real ro3 = rol * std::pow(pr3 / prl, 1.0 / gm);
    Real vx3 = vx2;

    Real ro5 = rol;
    Real pr5 = prl;
    Real vx5 = vxl;

    for (std::size_t i = 0; i < x.size(); ++i) {
      // Region 1
      if (x[i] > xm + vs * time) {
        ro[i] = ro1;
        vx[i] = vx1;
        pr[i] = pr1;
      }
      // Region 2
      else if (x[i] > xm + vc * time) {
        ro[i] = ro2;
        vx[i] = vx2;
        pr[i] = pr2;
      }
      // Region 3
      else if (x[i] >
               xm + ((gm + 1.0) / 2.0 * vc - csl - (gm - 1.0) / 2.0 * vxl) *
                        time) {
        ro[i] = ro3;
        vx[i] = vx3;
        pr[i] = pr3;
      }
      // Region 4
      else if (x[i] > xm + (vxl - csl) * time) {
        Real vx4 = 2.0 / (gm + 1.0) *
                   ((x[i] - xm) / time + csl + (gm - 1.0) / 2.0 * vxl);
        Real cs4 = csl - (gm - 1.0) / 2.0 * (vx4 - vxl);
        Real pr4 = prl * std::pow(cs4 / csl, alpha);
        Real ro4 = rol * std::pow(pr4 / prl, 1.0 / gm);

        ro[i] = ro4;
        pr[i] = pr4;
        vx[i] = vx4;
      } else {
        ro[i] = ro5;
        vx[i] = vx5;
        pr[i] = pr5;
      }
    }
  }
};
