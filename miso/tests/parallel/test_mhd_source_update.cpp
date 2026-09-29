#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_mhd_source_update_common.hpp"

using namespace test_source_update;

TEST_CASE("Heating source is scaled by dt (host)" *
          doctest::test_suite("mhd_source")) {
  Env env;
  for (const Real dt : {0.1, 0.01}) {
    CAPTURE(dt);
    check_heating<backend::Host>(dt);
  }
}
