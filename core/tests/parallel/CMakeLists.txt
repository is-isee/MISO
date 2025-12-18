# tests/parallel/CMakeLists.txt

if(USE_CUDA)
    set(source_suffix "cu")
else()
    set(source_suffix "cpp")
endif()
file(GLOB test_sources CONFIGURE_DEPENDS "test_*.${source_suffix}")

foreach(test_src ${test_sources})
    get_filename_component(test_name ${test_src} NAME_WE) # based on filename
    set(target_name ${test_name})

    add_executable(${target_name} ${test_src})

    target_include_directories(${target_name}
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(
        ${target_name} PRIVATE CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config/")
    target_link_libraries(${target_name} PRIVATE MISO::MISO doctest::doctest)

    set(mpi_num_procs 2)
    set(mpi_cmd ${CMAKE_SOURCE_DIR}/shared/mpiexec_with_env.sh
                ${MPIEXEC_NUMPROC_FLAG} ${mpi_num_procs})
    add_test(NAME ${test_name} COMMAND ${mpi_cmd} $<TARGET_FILE:${target_name}>)
endforeach()
