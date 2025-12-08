# problems/CMakeLists.txt
include(ExternalProject)

set(demo_names hd1d_shock_tube)
foreach(demo_name ${demo_names})
    set(demo_src ${CMAKE_CURRENT_SOURCE_DIR}/${demo_name})
    set(demo_bin ${CMAKE_BINARY_DIR}/_build_${demo_name})
    ExternalProject_Add(
        ${demo_name}
        SOURCE_DIR ${demo_src}
        BINARY_DIR ${demo_bin}
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DUSE_CUDA=${USE_CUDA}
        INSTALL_COMMAND "" # Without install
        STEP_TARGETS build)
endforeach()
