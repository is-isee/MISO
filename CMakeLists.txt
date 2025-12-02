cmake_minimum_required(VERSION 3.20)
project(MISO_superbuild NONE)
include(ExternalProject)

set(core_src ${CMAKE_SOURCE_DIR}/core)
set(demo_dir ${CMAKE_SOURCE_DIR}/demo)
set(core_bin ${CMAKE_BINARY_DIR}/_build_core)
set(STAGE ${CMAKE_BINARY_DIR}/_stage)

ExternalProject_Add(
    miso_core
    SOURCE_DIR ${CORE_SRC}
    BINARY_DIR ${CORE_BIN}
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_INSTALL_PREFIX=${STAGE} -DUSE_CUDA=ON # ここで切替
    INSTALL_DIR ${STAGE}
    STEP_TARGETS build install)

# cmake-format: off
# # 例：problem1 を独立ビルド
# ExternalProject_Add(
#     problem1
#     SOURCE_DIR ${PROB_DIR}/problem1
#     BINARY_DIR ${CMAKE_BINARY_DIR}/_build_problem1
#     CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#                -DCMAKE_PREFIX_PATH=${STAGE} # ← MISO を find_package させる
#     DEPENDS miso_core
#     STEP_TARGETS build)
# cmake-format: on
