#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#undef USE_CUDA
#include "test_model_mpi_common.hpp"

TEST_CASE("Test Context constructor and accessors") { run_test_model(); }
