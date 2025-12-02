# tests/parallel/CMakeLists.txt

if(USE_CUDA)
    # cmake-format: off
    set(mpi_opts
        "--bind-to" "none"
        "--mca" "pml" "ob1"
        "--mca" "btl" "tcp,self,vader"
        "--mca" "coll" "^hcoll"
        "--mca" "osc" "^ucx")
    # cmake-format: on
else()
    set(mpi_opts "")
endif()

if(USE_CUDA)
    set(source_suffix "cu")
else()
    set(source_suffix "cpp")
endif()
file(GLOB test_sources CONFIGURE_DEPENDS "test_*.${source_suffix}")

foreach(test_src ${test_sources})
    get_filename_component(test_name ${test_src} NAME_WE) # based on filename
    set(target_name ${test_name})

    add_executable(
        ${target_name}
        ${test_src}
        ${CMAKE_CURRENT_SOURCE_DIR}/custom_boundary_condition.${source_suffix})

    target_include_directories(${target_name}
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(
        ${target_name} PRIVATE CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config/")
    target_link_libraries(${target_name} PRIVATE MISO::MISO doctest::doctest)

    set(mpi_num_procs 2)
    set(mpi_cmd ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${mpi_num_procs})
    add_test(NAME ${test_name} COMMAND ${mpi_cmd} ${mpi_opts}
                                       $<TARGET_FILE:${target_name}>)
endforeach()
