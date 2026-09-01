# Run one command-line case and require its exit code and one complete line.
#
# cmake -DEXE=<path> -DCASE_ARGS=<semicolon-list> -DEXPECT_CODE=<int>
#       -DEXPECT_LINE=<line> -P run_cli_case.cmake
foreach(required EXE CASE_ARGS EXPECT_CODE EXPECT_LINE)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "run_cli_case.cmake requires -D${required}=...")
  endif()
endforeach()

execute_process(
  COMMAND "${EXE}" ${CASE_ARGS}
  OUTPUT_VARIABLE case_stdout
  ERROR_VARIABLE case_stderr
  RESULT_VARIABLE case_code)
set(case_output "${case_stdout}${case_stderr}")

if(NOT case_code EQUAL "${EXPECT_CODE}")
  message(FATAL_ERROR
    "expected exit ${EXPECT_CODE}, got '${case_code}'; output:\n"
    "${case_output}")
endif()
string(REPLACE "\r\n" "\n" case_output "${case_output}")
string(REPLACE "\r" "\n" case_output "${case_output}")
set(case_output_padded "\n${case_output}\n")
string(FIND "${case_output_padded}" "\n${EXPECT_LINE}\n" case_hit)
if(case_hit EQUAL -1)
  message(FATAL_ERROR
    "diagnostic line '${EXPECT_LINE}' not found in output:\n${case_output}")
endif()
message(STATUS
  "exit ${case_code} and diagnostic verified for: ${CASE_ARGS}")
