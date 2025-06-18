# tests/CMakeLists.txt
include(FetchContent)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.12
)
FetchContent_MakeAvailable(doctest)

file(GLOB TEST_SOURCES "*.cpp")
add_executable(unit_tests ${TEST_SOURCES})
target_include_directories(unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${doctest_SOURCE_DIR}
)

set_target_properties(unit_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )

target_compile_definitions(unit_tests PRIVATE CONFIG_DIR="${CMAKE_SOURCE_DIR}/tests/")
target_link_libraries(unit_tests PRIVATE MISOlib)
add_test(NAME run_doctest_tests COMMAND unit_tests)