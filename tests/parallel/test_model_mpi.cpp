#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_model_mpi_common.hpp"

TEST_CASE("Test Model constructor and accessors") {
  Model<Real> model = run_test_model();
}
