#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define USE_CUDA
#include "test_model_mpi_common.hpp"

TEST_CASE("Test Context GPU") { run_test_model(); }
