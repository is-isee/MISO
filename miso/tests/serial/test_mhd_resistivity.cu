#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_mhd_resistivity_common.hpp"

using namespace test_resistivity;

TEST_CASE("Resistivity GPU: same result as host" *
          doctest::test_suite("resistivity")) {
  const std::array<int, 3> sizes[3] = {{16, 1, 1}, {16, 12, 1}, {12, 10, 8}};
  for (const auto &n : sizes) {
    for (const auto &o : {orient_xyz, orient_yzx, orient_zxy}) {
      CAPTURE(n[0]);
      CAPTURE(n[1]);
      CAPTURE(n[2]);
      CAPTURE(o.axis[0]);
      Grid<Real, backend::Host> g = make_grid(n, o);
      mhd::Fields<Real, backend::Host> qq(g);
      Array3D<Real, backend::Host> eta(g.i_total, g.j_total, g.k_total);
      set_problem(g, o, b_smooth, eta_smooth, qq, eta);

      const Tendency t_host = tendency<backend::Host>(g, eta, qq);
      const Tendency t_cuda = tendency<backend::CUDA>(g, eta, qq);

      for (int i = g.i_margin; i < g.i_total - g.i_margin; ++i) {
        for (int j = g.j_margin; j < g.j_total - g.j_margin; ++j) {
          for (int k = g.k_margin; k < g.k_total - g.k_margin; ++k) {
            for (int m = 0; m < 3; ++m) {
              CHECK(t_cuda.db(m)(i, j, k) ==
                    doctest::Approx(t_host.db(m)(i, j, k)).epsilon(1e-12));
            }
            CHECK(t_cuda.de(i, j, k) ==
                  doctest::Approx(t_host.de(i, j, k)).epsilon(1e-12));
          }
        }
      }
    }
  }
}
