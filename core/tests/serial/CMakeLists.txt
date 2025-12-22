# tests/serial/CMakeLists.txt

function(miso_add_tests pattern suffix)
    file(GLOB test_sources CONFIGURE_DEPENDS ${pattern})
    foreach(test_src ${test_sources})
        get_filename_component(test_name ${test_src} NAME_WE)
        set(target_name serial_${test_name}_${suffix})

        add_executable(${target_name} ${test_src})
        target_link_libraries(${target_name} PRIVATE MISO::MISO
                                                     doctest::doctest)

        add_test(NAME ${target_name} COMMAND $<TARGET_FILE:${target_name}>)
    endforeach()
endfunction()

miso_add_tests("test_*.cpp" "cpp")
if(USE_CUDA)
    miso_add_tests("test_*.cu" "cu")
endif()
