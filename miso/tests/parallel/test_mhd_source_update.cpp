#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_mhd_source_update_common.hpp"

using namespace test_source_update;

TEST_CASE("Source terms through the MHD update (Host)" *
          doctest::test_suite("mhd_source")) {
  Env env;
  for (const Real dt : {0.1, 0.01}) {
    CAPTURE(dt);
    // heating only: the other terms are omitted in the Source
    check_heating<backend::Host>(dt);
    // all terms: each is added to its equation and scaled by dt
    check_all_terms<backend::Host>(dt);
  }
  // dt_limit() differs by rank; the global minimum is used
  check_dt_limit<backend::Host>();
}
