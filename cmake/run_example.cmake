include("${THRUST_SOURCE}/cmake/common_variables.cmake")

if (THRUST_FILECHECK_ENABLED)
  set(DATA_FILE "${THRUST_FILECHECK_DATA_PATH}/${THRUST_EXAMPLE}.filecheck")
  file(READ "${DATA_FILE}" CONTENTS)
  string(LENGTH "${CONTENTS}" LENGTH)
  message(${LENGTH})

  if (NOT ${LENGTH} EQUAL 0)
    set(FILECHECK_COMMAND
      COMMAND "${THRUST_FILECHECK}" "${THRUST_FILECHECK_DATA_PATH}/${THRUST_EXAMPLE}.filecheck")
  else ()
    set(CHECK_EMPTY_OUTPUT TRUE)
  endif ()
endif ()

execute_process(
  COMMAND "${THRUST_BINARY}"
  ${FILECHECK_COMMAND}
  RESULT_VARIABLE EXIT_CODE
  OUTPUT_VARIABLE STDOUT
  ERROR_VARIABLE STDERR
)

if (NOT "0" STREQUAL "${EXIT_CODE}")
  message(FATAL_ERROR "${THRUST_BINARY} failed (${EXIT_CODE}):\n${STDERR}")
endif ()

if (CHECK_EMPTY_OUTPUT)
  string(LENGTH "${OUTPUT_VARIABLE}" LENGTH)
  if (NOT ${LENGTH} EQUAL 0)
    message(FATAL_ERROR "${THRUST_BINARY}: output received, but not expected.")
  endif ()
endif ()
