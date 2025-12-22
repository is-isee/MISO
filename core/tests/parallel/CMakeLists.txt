# tests/parallel/CMakeLists.txt

function(miso_add_tests pattern suffix)
    file(GLOB test_sources CONFIGURE_DEPENDS ${pattern})
    foreach(test_src ${test_sources})
        get_filename_component(test_name ${test_src} NAME_WE)
        set(target_name ${test_name}_${suffix})

        add_executable(${target_name} ${test_src})

        target_include_directories(${target_name}
                                   PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_definitions(
            ${target_name}
            PRIVATE CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config/")
        target_link_libraries(${target_name} PRIVATE MISO::MISO
                                                     doctest::doctest)

        set(mpi_num_procs 2)
        set(mpi_cmd ${CMAKE_SOURCE_DIR}/shared/mpiexec_with_env.sh
                    ${MPIEXEC_NUMPROC_FLAG} ${mpi_num_procs})
        add_test(NAME ${target_name} COMMAND ${mpi_cmd}
                                             $<TARGET_FILE:${target_name}>)
    endforeach()
endfunction()

miso_add_tests("test_*.cpp" "cpp")
if(USE_CUDA)
    miso_add_tests("test_*.cu" "cu")
endif()
