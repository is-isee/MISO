#include "test_custom_boundary_condition_mpi_common.hpp"

TEST_CASE("Test Custom Boundary Condition GPU" *
          doctest::test_suite("custom_boundary_condition")) {
  run_custom_boundary_condition_mpi_tests();
}
