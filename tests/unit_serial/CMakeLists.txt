# tests/unit_serial/CMakeLists.txt

if(USE_CUDA)
    file(GLOB SERIAL_TEST_SOURCES CONFIGURE_DEPENDS "test_*.cu")
else()
    file(GLOB SERIAL_TEST_SOURCES CONFIGURE_DEPENDS "test_*.cpp")
endif()

add_executable(run_serial_tests ${SERIAL_TEST_SOURCES})
target_include_directories(run_serial_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(
    run_serial_tests PRIVATE CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config/")
target_link_libraries(run_serial_tests PRIVATE MISO::MISO doctest::doctest)
doctest_discover_tests(run_serial_tests)
