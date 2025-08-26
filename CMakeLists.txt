cmake_minimum_required(VERSION 3.18)
project(MISO)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose a type of build." FORCE)
enable_testing()

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
  list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
endif()

# --- external libraries: doctest ---
include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.12
)
FetchContent_MakeAvailable(doctest)
include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

# --- external libraries ---
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(yaml-cpp REQUIRED)

# --- shared library (src/*.cpp) ---
add_library(MISOlib INTERFACE)
target_include_directories(MISOlib INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(MISOlib INTERFACE
    yaml-cpp::yaml-cpp
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
)

# --- unit tests ---
add_subdirectory(tests)

# --- problems ---
add_subdirectory(problems)

# --- src ---
add_executable(MISO src/main.cpp)
