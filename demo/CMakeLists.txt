# problems/CMakeLists.txt

set(problem_targets
    hd1d_shock_tube
    hd2d_kelvin_helmholtz
    hd2d_rayleigh_taylor
    mhd1d_shock_tube
    mhd2d_vortex
    mhd3d_geomagnetosphere
    rt2d_searchlight)
if(USE_CUDA)
    list(APPEND problem_targets cuda_test)
endif()

foreach(problem_target ${problem_targets})
    add_subdirectory(${problem_target})
endforeach()

# add_custom_target(all_problems DEPENDS ${problem_targets})
