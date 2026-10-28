#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <miso/config.hpp>

TEST_CASE("Test Config parse_config_filepath" * doctest::test_suite("config")) {
  using namespace miso;

  const char *argv1[] = {"program", "--config", "config.yaml"};
  auto path1 = miso::parse_config_filepath(3, const_cast<char **>(argv1));
  REQUIRE(path1.has_value());
  REQUIRE(path1.value() == "config.yaml");

  const char *argv2[] = {"program", "--config=config2.yaml"};
  auto path2 = miso::parse_config_filepath(2, const_cast<char **>(argv2));
  REQUIRE(path2.has_value());
  REQUIRE(path2.value() == "config2.yaml");

  const char *argv3[] = {"program", "--other-arg", "value"};
  auto path3 = miso::parse_config_filepath(3, const_cast<char **>(argv3));
  REQUIRE_FALSE(path3.has_value());
}

TEST_CASE("Test Config defaults for rt" * doctest::test_suite("config")) {
  using namespace miso;

  YAML::Node user = YAML::Load("rt:\n  max_iters: 7\n");
  merge_yaml(user, YAML::Load(config_default_yaml));
  CHECK(user["rt"]["num_rays"].as<int>() == 24);
  CHECK(user["rt"]["tolerance"].as<double>() == 1.0e-4);
  CHECK(user["rt"]["max_iters"].as<int>() == 7);
}
