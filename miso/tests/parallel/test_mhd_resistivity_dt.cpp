#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_mhd_resistivity_dt_common.hpp"

using namespace test_resistivity_dt;

static Env env;

TEST_CASE("ResistiveSource::dt_limit with MPI (Host)" *
          doctest::test_suite("resistivity")) {
  check_dt_limit<backend::Host>();
}
