cmake_minimum_required(VERSION 3.18)
project(MISO_superbuild NONE)
include(ExternalProject)

option(USE_CUDA "Enable CUDA support" OFF)

set(miso_src ${CMAKE_SOURCE_DIR}/core)
set(miso_bin ${CMAKE_BINARY_DIR}/_build_miso)
ExternalProject_Add(
    miso
    SOURCE_DIR ${miso_src}
    BINARY_DIR ${miso_bin}
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DUSE_CUDA=${USE_CUDA}
    STEP_TARGETS build test
    INSTALL_COMMAND "" # Not to install doctest
    TEST_COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${miso_bin}
    # TEST_COMMAND "" # Uncomment to skip tests
)

# cmake-format: off
# set(demo_src ${CMAKE_SOURCE_DIR}/demo)
# set(demo_bin ${CMAKE_BINARY_DIR}/_build_demo)
# ExternalProject_Add(
#     demo
#     SOURCE_DIR ${demo_src}
#     BINARY_DIR ${demo_bin}
#     CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DUSE_CUDA=${USE_CUDA}
#                -DMISO_BIN=${miso_bin}
#     DEPENDS miso-build # Ensure MISO is built first
#     STEP_TARGETS build)
# cmake-format: on
