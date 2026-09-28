#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_mhd_resistivity_common.hpp"

using namespace test_resistivity;

TEST_CASE("Resistivity GPU: same result as host" *
          doctest::test_suite("resistivity")) {
  using Fields_d = mhd::Fields<Real, backend::CUDA>;
  using Array3D_d = Array3D<Real, backend::CUDA>;

  const std::array<int, 3> sizes[3] = {{16, 1, 1}, {16, 12, 1}, {12, 10, 8}};
  for (const auto &n : sizes) {
    for (const auto &o : {orient_xyz, orient_yzx, orient_zxy}) {
      CAPTURE(n[0]);
      CAPTURE(n[1]);
      CAPTURE(n[2]);
      CAPTURE(o.axis[0]);
      Grid_h g = make_grid(n, o);
      Fields_h qq(g);
      Array3D_h eta(g.i_total, g.j_total, g.k_total);
      set_problem(g, o, b_smooth, eta_smooth, qq, eta);

      // host
      Fields_h rslt_h(g);
      rslt_h.copy_from(qq);
      mhd::add_resistive_terms(backend::Host{}, Real(0.1), g.const_view(),
                               eta.const_view(), qq.const_view(), rslt_h.view());

      // device
      Grid<Real, backend::CUDA> g_d(g);
      Fields_d qq_d(g), rslt_d(g);
      Array3D_d eta_d(g.i_total, g.j_total, g.k_total);
      qq_d.copy_from(qq);
      rslt_d.copy_from(qq);
      eta_d.copy_from(eta);
      mhd::add_resistive_terms(backend::CUDA{}, Real(0.1), g_d.const_view(),
                               eta_d.const_view(), qq_d.const_view(),
                               rslt_d.view());
      MISO_CUDA_CHECK(cudaDeviceSynchronize());
      Fields_h rslt_dh(g);
      rslt_dh.copy_from(rslt_d);

      for (int idx = 0; idx < qq.ro.size(); ++idx) {
        CHECK(rslt_dh.bx[idx] == doctest::Approx(rslt_h.bx[idx]).epsilon(1e-13));
        CHECK(rslt_dh.by[idx] == doctest::Approx(rslt_h.by[idx]).epsilon(1e-13));
        CHECK(rslt_dh.bz[idx] == doctest::Approx(rslt_h.bz[idx]).epsilon(1e-13));
        CHECK(rslt_dh.ei[idx] == doctest::Approx(rslt_h.ei[idx]).epsilon(1e-13));
      }
    }
  }
}
