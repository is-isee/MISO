cmake_minimum_required(VERSION 3.18)
project(MISO_superbuild)
include(ExternalProject)

option(USE_CUDA "Enable CUDA support" OFF)

set(miso_src ${CMAKE_SOURCE_DIR}/core)
set(miso_bin ${CMAKE_BINARY_DIR}/_build_miso)
ExternalProject_Add(
    miso
    SOURCE_DIR ${miso_src}
    BINARY_DIR ${miso_bin}
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DUSE_CUDA=${USE_CUDA}
               -DENABLE_TESTING=ON
    STEP_TARGETS build test install
    INSTALL_COMMAND "" # Without install
    TEST_COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${miso_bin}
    # TEST_COMMAND "" # Uncomment to skip tests
)

add_subdirectory(demo)
