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

# add_custom_target(all_problems DEPENDS ${problem_targets})

# set(problem_targets hd1d_shock_tube hd2d_kelvin_helmholtz hd2d_rayleigh_taylor
# mhd1d_shock_tube mhd2d_vortex mhd3d_geomagnetosphere rt2d_searchlight)

# foreach(problem_target ${problem_targets}) add_subdirectory(${problem_target})
# endforeach()
