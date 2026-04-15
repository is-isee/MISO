#pragma once

#include <cassert>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include <yaml-cpp/yaml.h>

#include "config_defaults.hpp"
#include "env.hpp"
#include "utility.hpp"

namespace miso {

/// @brief Parse the path of configuration file from command line arguments
inline std::optional<std::string> parse_config_filepath(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);

    if (arg == "--config") {
      if (i + 1 >= argc)
        throw std::runtime_error("--config requires filepath");
      return std::string(argv[i + 1]);
    }

    constexpr std::string_view key = "--config=";
    if (arg.size() >= key.size() && arg.compare(0, key.size(), key) == 0) {
      return std::string(arg.substr(key.size()));
    }
  }
  return std::nullopt;
}

inline void merge_yaml(YAML::Node &target, const YAML::Node &defaults) {
  if (!defaults.IsMap()) {
    return;
  }

  if (!target || !target.IsMap()) {
    target = YAML::Node(YAML::NodeType::Map);
  }

  for (const auto &it : defaults) {
    const std::string key = it.first.as<std::string>();
    const YAML::Node &default_value = it.second;

    if (default_value.IsMap()) {
      YAML::Node child = target[key];
      merge_yaml(child, default_value);
      target[key] = child;
    } else {
      if (!target[key]) {
        target[key] = default_value;
      }
    }
  }
}

/// @brief Configuration class for MHD simulations
struct Config {
  /// @brief file path for YAML configuration file
  std::string load_filepath;
  /// @brief YAML object to hold the configuration information
  YAML::Node yaml_obj;
  /// @brief Directories for saving parent results
  std::string save_dir;

  Config(const std::string &load_filepath_) : load_filepath(load_filepath_) {
    std::string yaml_str;
    if (mpi::is_root()) {
      assert(!load_filepath.empty());
      if (!fs::exists(load_filepath)) {
        throw std::runtime_error("Config file not found: " + load_filepath);
      }
      yaml_obj = YAML::LoadFile(load_filepath);

      // output Endian type
      if (util::get_endian() == util::Endian::Little) {
        yaml_obj["data_type"]["Endian"] = "little";
      } else {
        yaml_obj["data_type"]["Endian"] = "big";
      }

      std::stringstream ss;
      ss << yaml_obj;
      yaml_str = ss.str();
    }

    int yaml_str_length = static_cast<int>(yaml_str.length());
    MPI_Bcast(&yaml_str_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!mpi::is_root()) {
      yaml_str.resize(yaml_str_length);
    }
    MPI_Bcast(yaml_str.data(), yaml_str_length, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (!mpi::is_root()) {
      yaml_obj = YAML::Load(yaml_str);
    }

    // read default config from config_defaults.hpp
    YAML::Node yaml_default_obj = YAML::Load(config_default_yaml);

    merge_yaml(yaml_obj, yaml_default_obj);

    fs::path config_path = fs::absolute(load_filepath);
    fs::path config_dir = config_path.parent_path();

    save_dir =
        (config_dir / yaml_obj["io"]["save_dir"].as<std::string>()).string();
    if (yaml_obj["io"]["enabled"].as<bool>()) {
      util::create_directories(save_dir);
    }
  }

  /// @brief Accessor for YAML configuration object
  const YAML::Node operator[](const std::string &key) const {
    return yaml_obj[key];
  }

  /// @brief  Save the configuration to a YAML file
  void save() const {
    if (mpi::is_root()) {
      if (!yaml_obj["io"]["enabled"].as<bool>()) {
        return;
      }
      std::string save_filepath = save_dir + "/config.yaml";
      std::ofstream ofs(save_filepath);
      if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file: " + save_filepath);
      }
      YAML::Emitter out;
      out << yaml_obj;
      ofs << out.c_str();
    }
  }
};

}  // namespace miso
