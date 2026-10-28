#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test_eos_table_common.hpp"

TEST_CASE("TableEOS computes pressure and sound speed in CUDA backend" *
          doctest::test_suite("eos_table")) {
  const auto path = write_table("eos_table_roen", true, "cu");
  check_gas_pressure_and_sound_speed<backend::CUDA>(path);
  fs::remove(path);
}
