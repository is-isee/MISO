# tests/CMakeLists.txt


if(USE_CUDA)
    set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cu)
    file(GLOB ALL_TEST_SOURCES CONFIGURE_DEPENDS "*.cu")
else()
    set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp)
    file(GLOB ALL_TEST_SOURCES CONFIGURE_DEPENDS "*.cpp")
endif()

# NON-MPI sources
set(NON_MPI_TEST_SOURCES ${ALL_TEST_SOURCES})
list(FILTER NON_MPI_TEST_SOURCES EXCLUDE REGEX ".*_mpi\\.(cpp|cu)$")
list(APPEND NON_MPI_TEST_SOURCES ${TEST_MAIN})


add_executable(non_mpi_unit_tests ${NON_MPI_TEST_SOURCES})
target_include_directories(non_mpi_unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(non_mpi_unit_tests
    PRIVATE CONFIG_DIR="${CMAKE_SOURCE_DIR}/tests/config/"
)
target_link_libraries(non_mpi_unit_tests PRIVATE
    MISOlib
    doctest::doctest)

doctest_discover_tests(non_mpi_unit_tests
    DISCOVERY_MODE TEST
    EXTRA_ARGS "--no-skip"
    EXCLUDE_LABEL "test_main"
)

# MPI sources
# ---------- MPI ----------
set(MPI_TEST_SOURCES ${ALL_TEST_SOURCES})
if(USE_CUDA)
    list(FILTER MPI_TEST_SOURCES INCLUDE REGEX ".*_mpi\\.cu$")
else()
    list(FILTER MPI_TEST_SOURCES INCLUDE REGEX ".*_mpi\\.cpp$")
endif()
# list(APPEND MPI_TEST_SOURCES ${TEST_MAIN})

add_custom_target(mpi_unit_tests)  # Umbrella target for MPI test

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

foreach(src ${MPI_TEST_SOURCES})
    get_filename_component(test_name ${src} NAME_WE)  # based on filename
    set(target_name ${test_name})

    if(USE_CUDA)
        add_executable(${target_name} ${src} ${TEST_MAIN} custom_boundary_condition.cu)
    else()
        add_executable(${target_name} ${src} ${TEST_MAIN} custom_boundary_condition.cpp)
    endif()
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(${target_name}
        PRIVATE CONFIG_DIR="${CMAKE_SOURCE_DIR}/tests/config/"
    )
    target_link_libraries(${target_name} PRIVATE
        MISOlib
        doctest::doctest)

    add_cuda_safe_mpi_test(${target_name} ${test_name} 2)

    # add_test(NAME ${test_name}
    #   COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
    #           $<TARGET_FILE:${target_name}> --no-skip
    # )
    # set_tests_properties(${test_name} PROPERTIES
    #   ENVIRONMENT "RUN_MPI=1"
    #   RUN_SERIAL TRUE
    #   LABELS "mpi"
    # )

    add_dependencies(mpi_unit_tests ${target_name})
endforeach()




# alias 
add_custom_target(unit_tests
    DEPENDS non_mpi_unit_tests mpi_unit_tests
)