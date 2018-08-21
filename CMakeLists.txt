cmake_minimum_required(VERSION 3.8)

project(Thrust CXX)

list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)

include(AppendOptionIfAvailable)

file(READ "thrust/version.h" THRUST_VERSION_HEADER)
string(REGEX MATCH "THRUST_VERSION ([0-9]+)" DUMMY ${THRUST_VERSION_HEADER})
set(THRUST_VERSION ${CMAKE_MATCH_1})
math(EXPR THRUST_VERSION_MAJOR "(${THRUST_VERSION} / 100000)")
math(EXPR THRUST_VERSION_MINOR "(${THRUST_VERSION} / 100) % 1000")
math(EXPR THRUST_VERSION_PATCH "${THRUST_VERSION} % 100")
set(
  THRUST_VERSION_STR
  "${THRUST_VERSION_MAJOR}.${THRUST_VERSION_MINOR}.${THRUST_VERSION_PATCH}"
)
message(STATUS "Thrust Version: ${THRUST_VERSION_STR}")

set(THRUST_HOST_BACKEND_OPTIONS CPP OMP TBB)
set(THRUST_HOST_BACKEND CPP CACHE STRING "The device backend to target.")
set_property(
  CACHE THRUST_HOST_BACKEND
  PROPERTY STRINGS ${THRUST_HOST_BACKEND_OPTIONS}
)
if (NOT THRUST_HOST_BACKEND IN_LIST THRUST_HOST_BACKEND_OPTIONS)
  message(
    FATAL_ERROR
    "THRUST_HOST_BACKEND must be one of ${THRUST_HOST_BACKEND_OPTIONS}"
  )
endif ()

set(THRUST_DEVICE_BACKEND_OPTIONS CUDA CPP OMP TBB)
set(THRUST_DEVICE_BACKEND CUDA CACHE STRING "The device backend to target.")
set_property(
  CACHE THRUST_DEVICE_BACKEND
  PROPERTY STRINGS ${THRUST_DEVICE_BACKEND_OPTIONS}
)
if (NOT THRUST_DEVICE_BACKEND IN_LIST THRUST_DEVICE_BACKEND_OPTIONS)
  message(
    FATAL_ERROR
    "THRUST_DEVICE_BACKEND must be one of ${THRUST_DEVICE_BACKEND_OPTIONS}"
  )
endif ()

if ("CUDA" STREQUAL "${THRUST_DEVICE_BACKEND}")
  enable_language(CUDA)
endif ()

if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 1700)
    message(FATAL_ERROR "This version of MSVC no longer supported.")
  endif ()
endif ()

if ("GNU" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.4)
    message(FATAL_ERROR "This version of GCC no longer supported.")
  endif ()
endif ()

if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # TODO Enable /Wall
  append_option_if_available(CXX "/WX" THRUST_OPTIONS_WARNINGS)

  # Disabled loss-of-data conversion warnings.
  # TODO Re-enable.
  append_option_if_available(CXX "/wd4244" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "/wd4267" THRUST_OPTIONS_WARNINGS)

  # Suppress numeric conversion-to-bool warnings.
  # TODO Re-enable.
  append_option_if_available(CXX "/wd4800" THRUST_OPTIONS_WARNINGS)

  # Disable warning about applying unary operator- to unsigned type.
  append_option_if_available(CXX "/wd4146" THRUST_OPTIONS_WARNINGS)
else ()
  append_option_if_available(CXX "-Werror" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Wall" THRUST_O:TIONS_WARNINGS)
  append_option_if_available(CXX "-Wextra" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Winit-self" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Woverloaded-virtual" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Wcast-qual" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Wno-cast-align" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Wno-long-long" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Wno-variadic-macros" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Wno-unused-function" THRUST_OPTIONS_WARNINGS)
endif ()

if ("GNU" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.5)
    # In GCC 4.4, the CUDA backend's kernel launch templates cause
    # impossible-to-decipher "'<anonymous>' is used uninitialized in this
    # function" warnings, so we disable uninitialized variable warnings.
    append_option_if_available(CXX "-Wno-uninitialized" THRUST_OPTIONS_WARNINGS)
  endif ()

  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 4.5)
    # This isn't available until GCC 4.3, and misfires on TMP code until
    # GCC 4.5.
    append_option_if_available(CXX "-Wlogical-op" THRUST_OPTIONS_WARNINGS)
  endif ()

  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.3)
    # GCC 7.3 complains about name mangling changes due to `noexcept`
    # becoming part of the type system; we don't care.
    append_option_if_available(CXX "-Wnoexcept-type" THRUST_OPTIONS_WARNINGS)
  endif ()
endif ()

if (("Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}") OR
    ("XL" STREQUAL "${CMAKE_CXX_COMPILER_ID}"))
  # xlC and Clang warn about unused parameters in uninstantiated templates.
  # This causes xlC to choke on the OMP backend, which is mostly #ifdef'd out
  # (and thus has unused parameters) when you aren't using it.
  append_option_if_available(CXX "-Wno-unused-parameters" THRUST_OPTIONS_WARNINGS)
endif ()
        
if ("Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # -Wunneeded-internal-declaration misfires in the unit test framework
  # on older versions of Clang.
  append_option_if_available(CXX "-Wno-unneeded-internal-declaration" THRUST_OPTIONS_WARNINGS)
endif ()
  
append_option_if_available(CUDA "-rdc=true" THRUST_OPTIONS_RDC)

set(THRUST_OPTIONS_DEBUG ${THRUST_OPTIONS_WARNINGS})
set(THRUST_OPTIONS_RELEASE ${THRUST_OPTIONS_WARNINGS})

include(CTest)
enable_testing()

list(APPEND THRUST_TESTFRAMEWORK_FILES testing/unittest/testframework.cu)
if ("CUDA" STREQUAL "${THRUST_DEVICE_BACKEND}")
  list(APPEND THRUST_TESTFRAMEWORK_FILES testing/unittest/cuda/testframework.cu)
endif ()

add_library(thrust_testframework STATIC ${THRUST_TESTFRAMEWORK_FILES})
target_include_directories(
  thrust_testframework
  PUBLIC ${PROJECT_SOURCE_DIR}
  PRIVATE ${PROJECT_SOURCE_DIR}/testing
)

list(APPEND THRUST_TEST_GLOBS testing/*.cu)
list(APPEND THRUST_TEST_GLOBS testing/*.cpp)

if     ("CUDA" STREQUAL "${THRUST_DEVICE_BACKEND}")
  list(APPEND THRUST_TEST_GLOBS testing/cuda/*.cu)
elseif ("OMP" STREQUAL "${THRUST_DEVICE_BACKEND}")
  list(APPEND THRUST_TEST_GLOBS testing/omp/*.cu)
  list(APPEND THRUST_TEST_GLOBS testing/omp/*.cpp)
endif ()

if (CMAKE_VERSION VERSION_LESS 3.12)
  file(
    GLOB THRUST_TESTS
    RELATIVE ${PROJECT_SOURCE_DIR}/testing
    ${THRUST_TEST_GLOBS}
    CONFIGURE_DEPENDS
  ) 
else ()
  file(
    GLOB THRUST_TESTS
    RELATIVE ${PROJECT_SOURCE_DIR}/testing
    ${THRUST_TEST_GLOBS}
  ) 
endif ()

foreach (THRUST_TEST_SOURCE IN LISTS THRUST_TESTS)
  # TODO: Per-test flags.

  get_filename_component(THRUST_TEST_CATEGORY ${THRUST_TEST_SOURCE} DIRECTORY)
  if (NOT ("" STREQUAL "${THRUST_TEST_CATEGORY}"))
    set(THRUST_TEST_CATEGORY "${THRUST_TEST_CATEGORY}.")
  endif () 

  get_filename_component(THRUST_TEST ${THRUST_TEST_SOURCE} NAME_WE)

  set(THRUST_TEST "thrust.test.${THRUST_TEST_CATEGORY}${THRUST_TEST}")
  set(THRUST_TEST_RDC "thrust.test.${THRUST_TEST_CATEGORY}rdc.${THRUST_TEST}")

  add_executable(
    ${THRUST_TEST}
    ${PROJECT_SOURCE_DIR}/testing/${THRUST_TEST_SOURCE}
  )

  add_executable(
    ${THRUST_TEST_RDC}
    ${PROJECT_SOURCE_DIR}/testing/${THRUST_TEST_SOURCE}
  )

  target_compile_options(${THRUST_TEST}
    PRIVATE "$<$<CONFIG:DEBUG>:${THRUST_OPTIONS_DEBUG}>"
            "$<$<CONFIG:RELEASE>:${THRUST_OPTIONS_RELEASE}>")

  target_compile_options(${THRUST_TEST_RDC}
    PRIVATE "$<$<CONFIG:DEBUG>:${THRUST_OPTIONS_DEBUG} ${THRUST_OPTIONS_RDC}>"
            "$<$<CONFIG:RELEASE>:${THRUST_OPTIONS_RELEASE} ${THRUST_OPTIONS_RDC}>")

  target_include_directories(
    ${THRUST_TEST}
    PUBLIC ${PROJECT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/testing
  )

  target_include_directories(
    ${THRUST_TEST_RDC}
    PUBLIC ${PROJECT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/testing
  )

  target_link_libraries(${THRUST_TEST}     thrust_testframework)
  target_link_libraries(${THRUST_TEST_RDC} thrust_testframework)

  add_test(run.${THRUST_TEST}     ${THRUST_TEST})
  add_test(run.${THRUST_TEST_RDC} ${THRUST_TEST_RDC})
endforeach ()

