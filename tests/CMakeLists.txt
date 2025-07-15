# tests/CMakeLists.txt

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "*.cpp")

add_executable(unit_tests ${TEST_SOURCES})
target_include_directories(unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(unit_tests
    PRIVATE CONFIG_DIR="${CMAKE_SOURCE_DIR}/tests/"
)
target_link_libraries(unit_tests PRIVATE
    MISOlib
    doctest::doctest)

doctest_discover_tests(unit_tests
    DISCOVERY_MODE TEST
    EXTRA_ARGS "--no-skip"
)
