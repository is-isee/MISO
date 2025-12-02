# tests/serial/CMakeLists.txt

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
    target_link_libraries(${target_name} PRIVATE MISO::MISO doctest::doctest)

    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${target_name}>)
endforeach()
