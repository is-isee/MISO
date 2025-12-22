#pragma once

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>

#include <mpi.h>
#include <yaml-cpp/yaml.h>

#include <miso/env.hpp>
#include <miso/types.hpp>
#include <miso/utility.hpp>

namespace miso {

/// @brief Create directories if they do not exist (only on root process)
void create_directories(const std::string &dir_path) {
  if (!mpi::is_root()) {
    return;
  }
  namespace fs = std::filesystem;
  fs::create_directories(dir_path);
}

/// @brief Configuration class for MHD simulations
struct Config {
  /// @brief file path for YAML configuration file
  std::string load_filepath;
  /// @brief YAML object to hold the configuration information
  YAML::Node yaml_obj;
  /// @brief Directories for saving parent results
  std::string save_dir;
  /// @brief Directories for saving time information
  std::string time_save_dir;
  /// @brief Directories for saving MHD results
  std::string mhd_save_dir;
  /// @brief Directories for saving MPI-related information
  std::string mpi_save_dir;

  Config(const std::string &load_filepath_)
      : load_filepath(load_filepath_), mpi_rt(mpi_env_) {
    std::string yaml_str;
    if (mpi::is_root()) {
      assert(!load_filepath.empty());
      if (!fs::exists(load_filepath)) {
        throw std::runtime_error("Config file not found: " + load_filepath);
      }
      yaml_obj = YAML::LoadFile(load_filepath);

      // output Real type
      if constexpr (std::is_same_v<Real, float>) {
        yaml_obj["data_type"]["Real"] = "float";
      } else if constexpr (std::is_same_v<Real, double>) {
        yaml_obj["data_type"]["Real"] = "double";
      } else {
        throw std::runtime_error("Unsupported Real type");
      }

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

    int yaml_str_length = yaml_str.length();
    MPI_Bcast(&yaml_str_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!mpi::is_root()) {
      yaml_str.resize(yaml_str_length);
    }
    MPI_Bcast(yaml_str.data(), yaml_str_length, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (!mpi::is_root()) {
      yaml_obj = YAML::Load(yaml_str);
    }

    fs::path config_path = fs::absolute(load_filepath);
    fs::path config_dir = config_path.parent_path();

    save_dir =
        (config_dir / yaml_obj["base"]["save_dir"].template as<std::string>())
            .string();
    create_directories(save_dir);
    mhd_save_dir =
        save_dir + yaml_obj["mhd"]["mhd_save_dir"].template as<std::string>();
    create_directories(mhd_save_dir);
  }

  /// @brief  Save the configuration to a YAML file
  void save() const {
    if (mpi::is_root()) {
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
