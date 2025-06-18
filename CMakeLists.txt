cmake_minimum_required(VERSION 3.10)
project(rmhdlab)

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
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)



# --- shared library (src/*.cpp) ---
add_library(rmhdlablib INTERFACE)
target_include_directories(rmhdlablib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(rmhdlablib
    INTERFACE
    yaml-cpp
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
)

add_custom_target(full_build
    DEPENDS rmhdlablib tests all_problems
)

# --- tests ---
add_subdirectory(tests EXCLUDE_FROM_ALL)
add_custom_target(unit_test DEPENDS unit_tests)

# --- problems ---
add_custom_target(problem DEPENDS all_problems)
add_subdirectory(problems EXCLUDE_FROM_ALL)
