get_filename_component(PROBLEM_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_executable(${PROBLEM_NAME} main.cu)
set_target_properties(${PROBLEM_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 ${CMAKE_BINARY_DIR})
target_link_libraries(${PROBLEM_NAME} MISO::MISO)

target_compile_definitions(${PROBLEM_NAME}
                           PRIVATE CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(
    run_${PROBLEM_NAME}
    COMMAND ${CMAKE_BINARY_DIR}/${PROBLEM_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(
    clean_data_${PROBLEM_NAME}
    COMMAND ${CMAKE_COMMAND} -E echo "Removing data* in ${PROBLEM_NAME}..."
    COMMAND /bin/sh -c "rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/data*"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

function(enable_always_assert_for target)
    # C/C++
    target_compile_options(
        ${target} PRIVATE $<$<COMPILE_LANGUAGE:C>:-UNDEBUG>
                          $<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG>)
    # CUDA（デバイス側にも、ホスト側コンパイラにも伝える）
    target_compile_options(
        ${target}
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG;-Xcompiler=-UNDEBUG>)
endfunction()

enable_always_assert_for(${PROBLEM_NAME})

target_include_directories(
    ${PROBLEM_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include
                            ${CMAKE_SOURCE_DIR}/problems/${PROBLEM_NAME})
