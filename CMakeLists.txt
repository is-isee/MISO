cmake_minimum_required(VERSION 3.18)
project(MISO)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE
    Release
    CACHE STRING "Choose a type of build." FORCE)

# --- CUDA オプションと条件付き有効化 ---
option(USE_CUDA "Disable CUDA support" OFF)
if(USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # --- GPUアーキテクチャの指定（L40S/4090等：Ada）---
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# --- macOS の場合 ---
if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/")
endif()

# --- external libraries ---
find_package(yaml-cpp REQUIRED)
find_package(MPI REQUIRED)

# --- shared library (src/*.cpp) ---
add_library(MISO INTERFACE)
add_library(MISO::MISO ALIAS MISO)
target_include_directories(MISO INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(MISO INTERFACE yaml-cpp::yaml-cpp MPI::MPI_CXX)

# --- unit tests ---
option(ENABLE_TESTING "Enable Testing" ON)
if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
