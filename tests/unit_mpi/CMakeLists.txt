# tests/unit_mpi/CMakeLists.txt

# Helper function to add CUDA-safe MPI flags when using CUDA
function(add_cuda_safe_mpi_test target test_name np)
    set(mpi_cmd ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np})
    if(USE_CUDA)
        list(APPEND mpi_cmd
            --bind-to none
            --mca pml ob1
            --mca btl tcp,self,vader
            --mca coll ^hcoll
            --mca osc ^ucx
        )
    endif()

    add_test(NAME ${test_name}
        COMMAND ${mpi_cmd} $<TARGET_FILE:${target}> --no-skip
    )

    set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "RUN_MPI=1"
        RUN_SERIAL TRUE
        LABELS "mpi"
    )
endfunction()

if(USE_CUDA)
    file(GLOB MPI_TEST_SOURCES CONFIGURE_DEPENDS "test_*.cu")
else()
    file(GLOB MPI_TEST_SOURCES CONFIGURE_DEPENDS "test_*.cpp")
endif()

foreach(test_src ${MPI_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)  # based on filename
    set(target_name ${test_name})

    if(USE_CUDA)
        add_executable(${target_name}
            ${test_src}
            ${TEST_MAIN}
            ${CMAKE_CURRENT_SOURCE_DIR}/custom_boundary_condition.cu
        )
        add_executable(${target_name}
            ${test_src}
            ${TEST_MAIN}
            ${CMAKE_CURRENT_SOURCE_DIR}/custom_boundary_condition.cu
            )
    else()
        add_executable(${target_name}
            ${test_src}
            ${TEST_MAIN}
            ${CMAKE_CURRENT_SOURCE_DIR}/custom_boundary_condition.cpp
        )
    endif()
    target_include_directories(${target_name}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_definitions(${target_name}
        PRIVATE CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config/"
    )
    target_link_libraries(${target_name} PRIVATE
        MISOlib
        doctest::doctest
    )
    add_cuda_safe_mpi_test(${target_name} ${test_name} 2)
    add_dependencies(mpi_unit_tests ${target_name})
endforeach()
