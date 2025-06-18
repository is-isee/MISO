cmake_minimum_required(VERSION 3.10)
project(MISO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
enable_testing()

# === CUDA オプションと条件付き有効化 ===
option(USE_CUDA "Enable CUDA support" ON)

if(USE_CUDA)
  enable_language(CUDA)
  add_definitions(-DUSE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # GPUアーキテクチャの指定（L40S/4090等：Ada）
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# --- external libraries ---
find_package(yaml-cpp REQUIRED)

if(APPLE)
  # Homebrew/OpenMP setting
  set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/include")
  set(OpenMP_C_LIB_NAMES "omp")
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/include")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY /opt/homebrew/lib/libomp.dylib)

  # link directories and rpath are explicitly defined
  link_directories(/opt/homebrew/lib)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "/opt/homebrew/lib")
endif()

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)



# --- shared library (src/*.cpp) ---
add_library(MISOlib INTERFACE)
target_include_directories(MISOlib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(MISOlib
    INTERFACE
    yaml-cpp
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
)

add_custom_target(full_build
    DEPENDS MISOlib tests all_problems
)

# --- tests ---
add_subdirectory(tests EXCLUDE_FROM_ALL)
add_custom_target(unit_test DEPENDS unit_tests)

# --- problems ---
add_custom_target(problem DEPENDS all_problems)
add_subdirectory(problems EXCLUDE_FROM_ALL)
