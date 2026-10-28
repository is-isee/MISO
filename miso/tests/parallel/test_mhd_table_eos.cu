#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_mhd_table_eos_common.hpp"

TEST_CASE("TableEOS through the MHD solver (CUDA)" *
          doctest::test_suite("eos_table")) {
  miso::Env env;
  test_mhd_table_eos::check_table_eos<miso::backend::CUDA>();
}
