#pragma once

#include <string>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <type_traits>
#include <mpi.h>
#include <yaml-cpp/yaml.h>
#include "mpi_manager.hpp"
#include "types.hpp"
#include "utility.hpp"

namespace fs = std::filesystem;

struct Config {

    std::string load_filepath;
    YAML::Node yaml_obj;
    std::string save_dir, time_save_dir, mhd_save_dir, mpi_save_dir;
    MPIManager<Real>& mpi;

    Config(const std::string& load_filepath_, MPIManager<Real>& mpi_)
        : load_filepath(load_filepath_), mpi(mpi_) {
            std::string yaml_str;
            if (mpi.myrank == 0) {
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

            if (mpi.myrank != 0) {
                yaml_str.resize(yaml_str_length);
            }
            MPI_Bcast(yaml_str.data(), yaml_str_length, MPI_CHAR, 0, MPI_COMM_WORLD);

            if (mpi.myrank != 0) {
                yaml_obj = YAML::Load(yaml_str);
            }

            fs::path config_path = fs::absolute(load_filepath);
            fs::path config_dir  = config_path.parent_path();

            save_dir = (config_dir / yaml_obj["base"]["save_dir"].as<std::string>()).string();
            time_save_dir = save_dir + yaml_obj["time"]["time_save_dir"].as<std::string>();
            mhd_save_dir  = save_dir + yaml_obj["mhd"]["mhd_save_dir"].as<std::string>();
            mpi_save_dir  = save_dir + yaml_obj["mpi"]["mpi_save_dir"].as<std::string>();
        }

    void create_save_directory_core(const std::string& directory) const {
        fs::path directory_fs(directory);
        if (!fs::exists(directory_fs)) {
            fs::create_directories(directory_fs);
        }
    }

    void create_save_directory() const {
        if (mpi.myrank == 0) {
            create_save_directory_core(save_dir);
            create_save_directory_core(time_save_dir);
            create_save_directory_core(mhd_save_dir);
            create_save_directory_core(mpi_save_dir);
        }
    }

    void save() const {
        if (mpi.myrank == 0) {
            std::string save_filepath = save_dir + "/config.yaml";
            std::ofstream ofs(save_filepath);
            if (!ofs.is_open()) {
                throw std::runtime_error("Failed to open file: " + save_filepath);
            }
            YAML::Emitter out;
            out << yaml_obj;
            ofs << out.c_str();
            ofs.close();
        }
    }
};