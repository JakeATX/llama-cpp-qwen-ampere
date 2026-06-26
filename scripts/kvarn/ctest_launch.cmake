set(_exe "$ENV{KVARN_CTEST_EXE}")
set(_args "$ENV{KVARN_CTEST_ARGS}")
set(_stdout "$ENV{KVARN_CTEST_STDOUT}")
set(_stderr "$ENV{KVARN_CTEST_STDERR}")
set(_workdir "$ENV{KVARN_CTEST_WORKDIR}")

if("${_exe}" STREQUAL "")
    message(FATAL_ERROR "KVARN_CTEST_EXE is required")
endif()
if("${_workdir}" STREQUAL "")
    set(_workdir "${CMAKE_CURRENT_LIST_DIR}/../..")
endif()
if("${_stdout}" STREQUAL "")
    set(_stdout "${_workdir}/ctest-launch.stdout.txt")
endif()
if("${_stderr}" STREQUAL "")
    set(_stderr "${_workdir}/ctest-launch.stderr.txt")
endif()

execute_process(
    COMMAND "${_exe}" ${_args}
    WORKING_DIRECTORY "${_workdir}"
    RESULT_VARIABLE _result
    OUTPUT_FILE "${_stdout}"
    ERROR_FILE "${_stderr}"
)

if(NOT _result EQUAL 0)
    message(FATAL_ERROR "command failed with exit code ${_result}; stdout=${_stdout}; stderr=${_stderr}")
endif()
