#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_rt_mhd_common.hpp"

TEST_CASE("Q_rad as a source term of the MHD solver (CUDA)" *
          doctest::test_suite("rt_heating")) {
  miso::Env env;
  test_rt_mhd::check_radiation<miso::backend::CUDA>();
}
