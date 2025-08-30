set(PROBLEM_EXECUTABLES)

foreach(PROBLEM_DIR
        hd_shock_tube_1d
        mhd_shock_tube_1d
        mhd_vortex_2d
        hd_kh_2d
        rt_searchlight_2d
        )
    add_subdirectory(${PROBLEM_DIR})
    list(APPEND PROBLEM_EXECUTABLES ${PROBLEM_DIR})
endforeach()

add_custom_target(all_problems DEPENDS ${PROBLEM_EXECUTABLES})
