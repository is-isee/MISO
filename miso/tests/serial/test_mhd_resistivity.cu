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
      const Tendency t_h = tendency_host(g, eta, qq);

      // Evaluate the source terms on the device
      Grid<Real, backend::CUDA> g_d(g);
      Fields_d qq_d(g);
      Array3D_d eta_d(g.i_total, g.j_total, g.k_total);
      qq_d.copy_from(qq);
      eta_d.copy_from(eta);
      const mhd::ResistiveSource<Real, backend::CUDA> src(g_d, eta_d, 0.5);

      Array3D_d dbx(g.i_total, g.j_total, g.k_total),
          dby(g.i_total, g.j_total, g.k_total),
          dbz(g.i_total, g.j_total, g.k_total),
          de(g.i_total, g.j_total, g.k_total);
      const auto q = qq_d.const_view();
      auto dbx_v = dbx.view(), dby_v = dby.view(), dbz_v = dbz.view(),
           de_v = de.view();
      Range3D range{{g.i_margin, g.i_total - g.i_margin},
                    {g.j_margin, g.j_total - g.j_margin},
                    {g.k_margin, g.k_total - g.k_margin}};
      for_each(
          backend::CUDA{}, range, MISO_LAMBDA(int i, int j, int k) {
            dbx_v(i, j, k) = src.bx(q, i, j, k);
            dby_v(i, j, k) = src.by(q, i, j, k);
            dbz_v(i, j, k) = src.bz(q, i, j, k);
            de_v(i, j, k) = src.ei(q, i, j, k);
          });
      MISO_CUDA_CHECK(cudaDeviceSynchronize());

      Tendency t_d(g);
      t_d.dbx.copy_from(dbx);
      t_d.dby.copy_from(dby);
      t_d.dbz.copy_from(dbz);
      t_d.de.copy_from(de);
      for (int i = g.i_margin; i < g.i_total - g.i_margin; ++i) {
        for (int j = g.j_margin; j < g.j_total - g.j_margin; ++j) {
          for (int k = g.k_margin; k < g.k_total - g.k_margin; ++k) {
            for (int m = 0; m < 3; ++m) {
              CHECK(t_d.db(m)(i, j, k) ==
                    doctest::Approx(t_h.db(m)(i, j, k)).epsilon(1e-12));
            }
            CHECK(t_d.de(i, j, k) ==
                  doctest::Approx(t_h.de(i, j, k)).epsilon(1e-12));
          }
        }
      }
    }
  }
}
