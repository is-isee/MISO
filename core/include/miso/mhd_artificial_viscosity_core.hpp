#pragma once

#include <miso/utility.hpp>

namespace miso {
namespace mhd {
namespace artificial_viscosity {

///@brief inline functions for generalized minmod limiter
template <typename Real>
__host__ __device__ inline Real dqq_eval(Real qq_dw, Real qq_cn, Real qq_up,
                                         Real ep) {
  Real dqq_dw = qq_cn - qq_dw;
  Real dqq_up = qq_up - qq_cn;
  Real dqq_cn = (qq_up - qq_dw) * 0.5;

  Real dqq_max = util::max3(ep * dqq_dw, ep * dqq_up, dqq_cn);
  Real dqq_min = util::min3(ep * dqq_dw, ep * dqq_up, dqq_cn);

  return util::min2(Real(0.0), dqq_max) + util::max2(Real(0.0), dqq_min);
}

///@brief inline function for flux core calculation
template <typename Real>
__host__ __device__ inline Real flux_core(Real qq_dw, Real qq_up, Real dqq_dw,
                                          Real dqq_up, Real cc, Real fh) {
  Real qql = qq_dw + 0.5 * dqq_dw;
  Real qqr = qq_up - 0.5 * dqq_up;
  Real dqq = qq_up - qq_dw;
  dqq = std::copysign(util::max2(std::abs(dqq), Real(1e-20)), dqq);

  Real ra = util::min2(Real(1.0), (qqr - qql) / dqq);
  Real pp = (Real(0.5) + std::copysign(Real(0.5), ra)) *
            util::max2(Real(0.0), Real(1.0) + fh * (ra - Real(1.0)));

  return -0.5 * cc * pp * (qqr - qql);
}

}  // namespace artificial_viscosity
}  // namespace mhd
}  // namespace miso
