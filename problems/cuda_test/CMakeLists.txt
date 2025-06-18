get_filename_component(PROBLEM_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_executable(${PROBLEM_NAME} main.cu)
set_target_properties(${PROBLEM_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(${PROBLEM_NAME} MISOlib)

target_compile_definitions(${PROBLEM_NAME} PRIVATE CONFIG_DIR="${CMAKE_SOURCE_DIR}/problems/${PROBLEM_NAME}/")

add_custom_target(run_${PROBLEM_NAME}
  COMMAND ${CMAKE_BINARY_DIR}/${PROBLEM_NAME}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/problems/${PROBLEM_NAME}
)

add_custom_target(clean_data_${PROBLEM_NAME}
  COMMAND ${CMAKE_COMMAND} -E echo "Removing data* in ${PROBLEM_NAME}..."
  COMMAND /bin/sh -c "rm -rf ${CMAKE_SOURCE_DIR}/problems/${PROBLEM_NAME}/data*"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/problems/${PROBLEM_NAME}
)