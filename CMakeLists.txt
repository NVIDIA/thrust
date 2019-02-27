cmake_minimum_required(VERSION 3.8)

project(Thrust CXX)

set(THRUST_SOURCE ${CMAKE_SOURCE_DIR})
include(cmake/common_variables.cmake)

if ("" STREQUAL "${CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)

  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel
  )
endif ()

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.12)
  set(CMAKE_CONFIGURE_DEPENDS CONFIGURE_DEPENDS)
endif ()

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

set(THRUST_HOST_SYSTEM_OPTIONS CPP OMP TBB)
set(THRUST_HOST_SYSTEM CPP CACHE STRING "The device backend to target.")
set_property(
  CACHE THRUST_HOST_SYSTEM
  PROPERTY STRINGS ${THRUST_HOST_SYSTEM_OPTIONS}
)
if (NOT THRUST_HOST_SYSTEM IN_LIST THRUST_HOST_SYSTEM_OPTIONS)
  message(
    FATAL_ERROR
    "THRUST_HOST_SYSTEM must be one of ${THRUST_HOST_SYSTEM_OPTIONS}"
  )
endif ()

add_definitions(-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_${THRUST_HOST_SYSTEM})

set(THRUST_DEVICE_SYSTEM_OPTIONS CUDA CPP OMP TBB)
set(THRUST_DEVICE_SYSTEM CUDA CACHE STRING "The device backend to target.")
set_property(
  CACHE THRUST_DEVICE_SYSTEM
  PROPERTY STRINGS ${THRUST_DEVICE_SYSTEM_OPTIONS}
)
if (NOT THRUST_DEVICE_SYSTEM IN_LIST THRUST_DEVICE_SYSTEM_OPTIONS)
  message(
    FATAL_ERROR
    "THRUST_DEVICE_SYSTEM must be one of ${THRUST_DEVICE_SYSTEM_OPTIONS}"
  )
endif ()

add_definitions(-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_${THRUST_DEVICE_SYSTEM})

# Please note this also sets the default for the CUDA C++ version; see the comment below.
set(CMAKE_CXX_STANDARD 11 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_EXTENSIONS OFF)

message("-- C++ Standard version: ${CMAKE_CXX_STANDARD}")

if ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  if (NOT "${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "")
    unset(CMAKE_CUDA_HOST_COMPILER CACHE)
    message(FATAL_ERROR "Thrust tests and examples require the C++ compiler"
        " and the CUDA host compiler to be the same; to set this compiler, please"
        " use the CMAKE_CXX_COMPILER variable, not the CMAKE_CUDA_HOST_COMPILER"
        " variable.")
  endif ()
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

  enable_language(CUDA)

  # Force CUDA C++ standard to be the same as the C++ standard used.
  #
  # Now, CMake is unaligned with reality on standard versions: https://gitlab.kitware.com/cmake/cmake/issues/18597
  # which means that using standard CMake methods, it's impossible to actually sync the CXX and CUDA versions for pre-11
  # versions of C++; CUDA accepts 98 but translates that to 03, while CXX doesn't accept 03 (and doesn't translate that to 03).
  # In case this gives You, dear user, any trouble, please escalate the above CMake bug, so we can support reality properly.
  if (DEFINED CMAKE_CUDA_STANDARD)
      message(WARNING "You've set CMAKE_CUDA_STANDARD; please note that this variable is ignored, and CMAKE_CXX_STANDARD"
          " is used as the C++ standard version for both C++ and CUDA.")
  endif()
  unset(CMAKE_CUDA_STANDARD CACHE)
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})

  set(THRUST_HIGHEST_COMPUTE_ARCH 75)
  set(THRUST_KNOWN_COMPUTE_ARCHS 30 32 35 50 52 53 60 61 62 70 72 75)

  option(THRUST_DISABLE_ARCH_BY_DEFAULT "If ON, then all CUDA architectures are disabled on the initial CMake run." OFF)
  set(OPTION_INIT ON)
  if (THRUST_DISABLE_ARCH_BY_DEFAULT)
    set(OPTION_INIT OFF)
  endif ()

  if (NOT ${THRUST_HIGHEST_COMPUTE_ARCH} IN_LIST THRUST_KNOWN_COMPUTE_ARCHS)
    message(FATAL_ERROR "When changing the highest compute version, don't forget to add it to the list!")
  endif ()

  foreach (COMPUTE_ARCH IN LISTS THRUST_KNOWN_COMPUTE_ARCHS)
    option(THRUST_ENABLE_COMPUTE_${COMPUTE_ARCH} "Enable code generation for tests for sm_${COMPUTE_ARCH}" ${OPTION_INIT})
    if (THRUST_ENABLE_COMPUTE_${COMPUTE_ARCH})
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${COMPUTE_ARCH},code=sm_${COMPUTE_ARCH}")
      set(COMPUTE_MESSAGE "${COMPUTE_MESSAGE} sm_${COMPUTE_ARCH}")
    endif ()
  endforeach ()

  option(THRUST_ENABLE_COMPUTE_FUTURE "Enable code generation for tests for compute_${THRUST_HIGHEST_COMPUTE_ARCH}" ${OPTION_INIT})
  if (THRUST_ENABLE_COMPUTE_FUTURE)
    set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${THRUST_HIGHEST_COMPUTE_ARCH},code=compute_${THRUST_HIGHEST_COMPUTE_ARCH}")
    set(COMPUTE_MESSAGE "${COMPUTE_MESSAGE} compute_${THRUST_HIGHEST_COMPUTE_ARCH}")
  endif ()

  message("-- Enabled CUDA architectures:${COMPUTE_MESSAGE}")
endif ()

if ("OMP" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  find_package(OpenMP REQUIRED)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()
endif ()

if ("TBB" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(TBB tbb REQUIRED)
  if (TBB_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TDD_CFLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TDD_CFLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${TBB_LD_FLAGS}")
    set (THRUST_ADDITIONAL_LIBRARIES "${TBB_LIBRARIES}")
  endif ()

  # There's a ton of these in the TBB backend, even though the code is correct.
  # TODO: silence these warnings in code instead
  append_option_if_available("-Wno-unused-parameter" THRUST_CXX_WARNINGS)
endif ()

if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 1900)
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
  append_option_if_available("/WX" THRUST_CXX_WARNINGS)

  # Disabled loss-of-data conversion warnings.
  # TODO Re-enable.
  append_option_if_available("/wd4244" THRUST_CXX_WARNINGS)
  append_option_if_available("/wd4267" THRUST_CXX_WARNINGS)

  # Suppress numeric conversion-to-bool warnings.
  # TODO Re-enable.
  append_option_if_available("/wd4800" THRUST_CXX_WARNINGS)

  # Disable warning about applying unary operator- to unsigned type.
  append_option_if_available("/wd4146" THRUST_CXX_WARNINGS)

  set(THRUST_TREAT_FILE_AS_CXX "/TP")
else ()
  append_option_if_available("-Werror" THRUST_CXX_WARNINGS)
  append_option_if_available("-Wall" THRUST_CXX_WARNINGS)
  append_option_if_available("-Wextra" THRUST_CXX_WARNINGS)
  append_option_if_available("-Winit-self" THRUST_CXX_WARNINGS)
  append_option_if_available("-Woverloaded-virtual" THRUST_CXX_WARNINGS)
  append_option_if_available("-Wcast-qual" THRUST_CXX_WARNINGS)
  append_option_if_available("-Wno-cast-align" THRUST_CXX_WARNINGS)
  append_option_if_available("-Wno-long-long" THRUST_CXX_WARNINGS)
  append_option_if_available("-Wno-variadic-macros" THRUST_CXX_WARNINGS)
  append_option_if_available("-Wno-unused-function" THRUST_CXX_WARNINGS)
  append_option_if_available("-Wno-unused-variable" THRUST_CXX_WARNINGS)

  set(THRUST_TREAT_FILE_AS_CXX "-x c++")
endif ()

if ("GNU" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.5)
    # In GCC 4.4, the CUDA backend's kernel launch templates cause
    # impossible-to-decipher "'<anonymous>' is used uninitialized in this
    # function" warnings, so we disable uninitialized variable warnings.
    append_option_if_available("-Wno-uninitialized" THRUST_CXX_WARNINGS)
  endif ()

  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 4.5)
    # This isn't available until GCC 4.3, and misfires on TMP code until
    # GCC 4.5.
    append_option_if_available("-Wlogical-op" THRUST_CXX_WARNINGS)
  endif ()

  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.3)
    # GCC 7.3 complains about name mangling changes due to `noexcept`
    # becoming part of the type system; we don't care.
    append_option_if_available("-Wno-noexcept-type" THRUST_CXX_WARNINGS)
  endif ()

  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.1 AND CMAKE_CXX_STANDARD EQUAL 98)
    # thrust::complex can't really be made trivially copyable in pre-11.
    # Disable a warning about a non-trivially-copyable type being memmoved that was added to GCC 8.
    append_option_if_available("-Wno-class-memaccess" THRUST_CXX_WARNINGS)
  endif ()
endif ()

if (("Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}") OR
    ("XL" STREQUAL "${CMAKE_CXX_COMPILER_ID}"))
  # xlC and Clang warn about unused parameters in uninstantiated templates.
  # This causes xlC to choke on the OMP backend, which is mostly #ifdef'd out
  # (and thus has unused parameters) when you aren't using it.
  append_option_if_available("-Wno-unused-parameters" THRUST_CXX_WARNINGS)
endif ()

if ("Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # -Wunneeded-internal-declaration misfires in the unit test framework
  # on older versions of Clang.
  append_option_if_available("-Wno-unneeded-internal-declaration" THRUST_CXX_WARNINGS)
endif ()

foreach (CXX_OPTION IN LISTS THRUST_CXX_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_OPTION}")
endforeach ()

if ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  foreach (CXX_OPTION IN LISTS THRUST_CXX_WARNINGS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${CXX_OPTION}")
  endforeach ()
endif ()

# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.
list(APPEND THRUST_HEADER_GLOBS thrust/*.h)
list(APPEND THRUST_HEADER_EXCLUDE_SYSTEMS_GLOBS thrust/system/*/*)

string(TOLOWER ${THRUST_HOST_SYSTEM} THRUST_HOST_SYSTEM_LOWERCASE)
list(APPEND THRUST_HEADER_SYSTEMS_GLOBS thrust/system/${THRUST_HOST_SYSTEM_LOWERCASE}/*)

string(TOLOWER ${THRUST_DEVICE_SYSTEM} THRUST_DEVICE_SYSTEM_LOWERCASE)
list(APPEND THRUST_HEADER_SYSTEMS_GLOBS thrust/system/${THRUST_DEVICE_SYSTEM_LOWERCASE}/*)

list(APPEND THRUST_HEADER_EXCLUDE_DETAILS_GLOBS thrust/detail/*)
list(APPEND THRUST_HEADER_EXCLUDE_DETAILS_GLOBS thrust/*/detail/*)
list(APPEND THRUST_HEADER_EXCLUDE_DETAILS_GLOBS thrust/*/*/detail/*)

# Get all .h files...
file(
  GLOB_RECURSE THRUST_HEADERS
  RELATIVE ${PROJECT_SOURCE_DIR}/thrust
  ${CMAKE_CONFIGURE_DEPENDS}
  ${THRUST_HEADER_GLOBS}
)

# ...then remove all system specific headers...
file(
  GLOB_RECURSE THRUST_HEADER_EXCLUDE_SYSTEMS
  RELATIVE ${PROJECT_SOURCE_DIR}/thrust
  ${CMAKE_CONFIGURE_DEPENDS}
  ${THRUST_HEADER_EXCLUDE_SYSTEMS_GLOBS}
)
list(REMOVE_ITEM THRUST_HEADERS ${THRUST_HEADER_EXCLUDE_SYSTEMS})

# ...then add all headers specific to the selected host and device systems back again...
file(
  GLOB_RECURSE THRUST_SYSTEMS_HEADERS
  RELATIVE ${PROJECT_SOURCE_DIR}/thrust
  ${CMAKE_CONFIGURE_DEPENDS}
  ${THRUST_HEADER_SYSTEMS_GLOBS}
)
list(APPEND THRUST_HEADERS ${THRUST_SYSTEMS_HEADERS})

# ...and remove all the detail headers (also removing the detail headers from the selected systems).
file(
  GLOB_RECURSE THRUST_HEADER_EXCLUDE_DETAILS
  RELATIVE ${PROJECT_SOURCE_DIR}/thrust
  ${CMAKE_CONFIGURE_DEPENDS}
  ${THRUST_HEADER_EXCLUDE_DETAILS_GLOBS}
)
list(REMOVE_ITEM THRUST_HEADERS ${THRUST_HEADER_EXCLUDE_DETAILS})

# List of headers that aren't implemented for all backends, but are implemented for CUDA.
set(THRUST_PARTIALLY_IMPLEMENTED_HEADERS_CUDA
  async/copy.h
  async/for_each.h
  async/reduce.h
  async/sort.h
  async/transform.h
  event.h
  future.h
)

# List of headers that aren't implemented for all backends, but are implemented for CPP.
set(THRUST_PARTIALLY_IMPLEMENTED_HEADERS_CPP
)

# List of headers that aren't implemented for all backends, but are implemented for TBB.
set(THRUST_PARTIALLY_IMPLEMENTED_HEADERS_TBB
)

# List of headers that aren't implemented for all backends, but are implemented for OMP.
set(THRUST_PARTIALLY_IMPLEMENTED_HEADERS_OMP
)

# List of all partially implemented headers.
set(THRUST_PARTIALLY_IMPLEMENTED_HEADERS
  emptylistguard
  ${THRUST_PARTIALLY_IMPLEMENTED_HEADERS_CUDA}
  ${THRUST_PARTIALLY_IMPLEMENTED_HEADERS_CPP}
  ${THRUST_PARTIALLY_IMPLEMENTED_HEADERS_TBB}
  ${THRUST_PARTIALLY_IMPLEMENTED_HEADERS_OMP}
)

list(REMOVE_DUPLICATES THRUST_PARTIALLY_IMPLEMENTED_HEADERS)

foreach (THRUST_HEADER IN LISTS THRUST_HEADERS)
  if ("${THRUST_HEADER}" IN_LIST THRUST_PARTIALLY_IMPLEMENTED_HEADERS)
    # This header is partially implemented on _some_ backends...
    if (NOT "${THRUST_HEADER}" IN_LIST THRUST_PARTIALLY_IMPLEMENTED_HEADERS_${THRUST_DEVICE_SYSTEM})
      # ...but not on the selected one.
      continue()
    endif ()
  endif ()

  set(THRUST_HEADER_TEST_EXT .cpp)
  if ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
    set(THRUST_HEADER_TEST_EXT .cu)
  endif ()

  set(SOURCE_NAME headers/${THRUST_HEADER}${THRUST_HEADER_TEST_EXT})
  configure_file(cmake/header_test.in ${SOURCE_NAME})

  list(APPEND THRUST_HEADER_TEST_SOURCES ${SOURCE_NAME})
endforeach ()

add_library(header-test OBJECT ${THRUST_HEADER_TEST_SOURCES})
target_include_directories(
  header-test
  PUBLIC ${PROJECT_SOURCE_DIR}
)

include(CTest)
enable_testing()

# Handle tests.

option(THRUST_ENABLE_TESTS_WITH_RDC "Also build all tests with RDC." OFF)

set(THRUST_TEST_RUN_ARGUMENTS
  -DTHRUST_SOURCE=${CMAKE_SOURCE_DIR}
  -P "${CMAKE_SOURCE_DIR}/cmake/run_test.cmake")

list(APPEND THRUST_TESTFRAMEWORK_FILES testing/unittest/testframework.cu)
if ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  list(APPEND THRUST_TESTFRAMEWORK_FILES testing/unittest/cuda/testframework.cu)
else ()
  # When CUDA is disabled, explain to CMake that testframework.cu is actually a C++ file.
  set_source_files_properties(testing/unittest/testframework.cu
    PROPERTIES
      LANGUAGE CXX
      COMPILE_FLAGS "${THRUST_TREAT_FILE_AS_CXX}")
endif ()

add_library(thrust_testframework STATIC ${THRUST_TESTFRAMEWORK_FILES})
target_include_directories(
  thrust_testframework
  PUBLIC ${PROJECT_SOURCE_DIR}
  PRIVATE ${PROJECT_SOURCE_DIR}/testing
)

list(APPEND THRUST_TEST_GLOBS testing/*.cu)
list(APPEND THRUST_TEST_GLOBS testing/*.cpp)

if     ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  list(APPEND THRUST_TEST_GLOBS testing/cuda/*.cu)
elseif ("OMP" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  list(APPEND THRUST_TEST_GLOBS testing/omp/*.cu)
  list(APPEND THRUST_TEST_GLOBS testing/omp/*.cpp)
endif ()

file(
  GLOB THRUST_TESTS
  RELATIVE ${PROJECT_SOURCE_DIR}/testing
  ${CMAKE_CONFIGURE_DEPENDS}
  ${THRUST_TEST_GLOBS}
)

# List of tests that aren't implemented for all backends, but are implemented for CUDA.
set(THRUST_PARTIALLY_IMPLEMENTED_CUDA
    async_copy
    async_for_each
    async_reduce
    async_reduce_into
    async_sort
    async_transform
    event
    future
)

# List of tests that aren't implemented for all backends, but are implemented for CPP.
set(THRUST_PARTIALLY_IMPLEMENTED_CPP
)

# List of tests that aren't implemented for all backends, but are implemented for TBB.
set(THRUST_PARTIALLY_IMPLEMENTED_TBB
)

# List of tests that aren't implemented for all backends, but are implemented for OMP.
set(THRUST_PARTIALLY_IMPLEMENTED_OMP
)

# List of all partially implemented tests.
set(THRUST_PARTIALLY_IMPLEMENTED
  ${THRUST_PARTIALLY_IMPLEMENTED_CUDA}
  ${THRUST_PARTIALLY_IMPLEMENTED_CPP}
  ${THRUST_PARTIALLY_IMPLEMENTED_TBB}
  ${THRUST_PARTIALLY_IMPLEMENTED_OMP}
)

if ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  if (14 EQUAL ${CMAKE_CXX_STANDARD})
    # Temporarily disable until NVBug 2492786 is fixed.
    list(APPEND THRUST_PARTIALLY_IMPLEMENTED tuple_algorithms)
  endif()
endif ()

list(REMOVE_DUPLICATES THRUST_PARTIALLY_IMPLEMENTED)

foreach (THRUST_TEST_SOURCE IN LISTS THRUST_TESTS)
  # TODO: Per-test flags.

  set(THRUST_TEST_CREATION_ADDITIONAL)
  set(THRUST_TEST_ADD_TO_CTEST ON)

  get_filename_component(THRUST_TEST_CATEGORY ${THRUST_TEST_SOURCE} DIRECTORY)
  if (NOT ("" STREQUAL "${THRUST_TEST_CATEGORY}"))
    set(THRUST_TEST_CATEGORY "${THRUST_TEST_CATEGORY}.")
  endif ()

  get_filename_component(THRUST_TEST_NAME ${THRUST_TEST_SOURCE} NAME_WE)

  if ("${THRUST_TEST_NAME}" IN_LIST THRUST_PARTIALLY_IMPLEMENTED)
    # This test is partially implemented on _some_ backends...
    if (NOT "${THRUST_TEST_NAME}" IN_LIST THRUST_PARTIALLY_IMPLEMENTED_${THRUST_DEVICE_SYSTEM})
      # ...but not on the selected one.
      set(THRUST_TEST_CREATION_ADDITIONAL EXCLUDE_FROM_ALL)
      set(THRUST_TEST_ADD_TO_CTEST OFF)
    endif ()
  endif ()

  set(THRUST_TEST "thrust.test.${THRUST_TEST_CATEGORY}${THRUST_TEST_NAME}")

  if (NOT "CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
    # Test files are generally .cu; if CUDA is not enabled, CMake doesn't know what to
    # do with them. But since they are pretty much just C++, we can compile them with
    # non-nvcc C++ compilers... but we need to tell CMake that they are, in fact, just C++.
    set_source_files_properties(${PROJECT_SOURCE_DIR}/testing/${THRUST_TEST_SOURCE}
      PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "${THRUST_TREAT_FILE_AS_CXX}")
  endif ()

  add_executable(
    ${THRUST_TEST}
    ${THRUST_TEST_CREATION_ADDITIONAL}
    # THRUST_TEST_CREATION_ADDITIONAL is actually a CMake keyword (sometimes).
    ${PROJECT_SOURCE_DIR}/testing/${THRUST_TEST_SOURCE}
  )

  target_include_directories(
    ${THRUST_TEST}
    PUBLIC ${PROJECT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/testing
  )

  target_link_libraries(${THRUST_TEST}
    thrust_testframework
    ${THRUST_ADDITIONAL_LIBRARIES})

  if (THRUST_TEST_ADD_TO_CTEST)
    add_test(NAME ${THRUST_TEST}
      COMMAND ${CMAKE_COMMAND}
        -DTHRUST_BINARY=$<TARGET_FILE:${THRUST_TEST}>
        ${THRUST_TEST_RUN_ARGUMENTS})
  endif ()

  if ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}" AND THRUST_ENABLE_TESTS_WITH_RDC)
    set(THRUST_TEST_RDC "thrust.test.${THRUST_TEST_CATEGORY}rdc.${THRUST_TEST_NAME}")

    add_executable(
      ${THRUST_TEST_RDC}
      ${THRUST_TEST_CREATION_ADDITIONAL}
      # THRUST_TEST_CREATION_ADDITIONAL is actually a CMake keyword (sometimes).
      ${PROJECT_SOURCE_DIR}/testing/${THRUST_TEST_SOURCE}
    )

    target_include_directories(
      ${THRUST_TEST_RDC}
      PUBLIC ${PROJECT_SOURCE_DIR}
      PRIVATE ${PROJECT_SOURCE_DIR}/testing
    )

    target_link_libraries(${THRUST_TEST_RDC}
      thrust_testframework
      ${THRUST_ADDITIONAL_LIBRARIES})

    set_target_properties(${THRUST_TEST_RDC}
      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    if (THRUST_TEST_ADD_TO_CTEST)
      add_test(NAME ${THRUST_TEST_RDC}
        COMMAND ${CMAKE_COMMAND}
          -DTHRUST_BINARY=$<TARGET_FILE:${THRUST_TEST_RDC}>
          ${THRUST_TEST_RUN_ARGUMENTS})
    endif ()
  endif ()
endforeach ()

# Handle examples.

option(THRUST_EXAMPLE_FILECHECK_PATH "Path to the LLVM FileCheck utility." "")
option(THRUST_ENABLE_EXAMPLES_WITH_RDC "Also build all examples with RDC." OFF)

set(THRUST_EXAMPLE_FILECHECK_ENABLED OFF)
if (NOT "" STREQUAL "${THRUST_EXAMPLE_FILECHECK_PATH}")
  execute_process(
    COMMAND "${THRUST_EXAMPLE_FILECHECK_PATH}" "${THRUST_FILECHECK_DATA_PATH}/thrust.sanity.filecheck"
    INPUT_FILE "${CMAKE_SOURCE_DIR}/cmake/sanity"
    RESULT_VARIABLE THRUST_FILECHECK_RESULT
  )

  if ("0" STREQUAL "${THRUST_FILECHECK_RESULT}")
    set(THRUST_EXAMPLE_FILECHECK_ENABLED ON)
    message("-- FileCheck enabled: ${THRUST_EXAMPLE_FILECHECK_PATH}")
  endif ()
endif ()

list(APPEND THRUST_EXAMPLE_GLOBS examples/*.cu)
list(APPEND THRUST_EXAMPLE_GLOBS examples/*.cpp)

if     ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  list(APPEND THRUST_EXAMPLE_GLOBS examples/cuda/*.cu)
elseif ("OMP" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  list(APPEND THRUST_EXAMPLE_GLOBS examples/omp/*.cu)
  list(APPEND THRUST_EXAMPLE_GLOBS examples/omp/*.cpp)
endif ()

if (CMAKE_VERSION VERSION_LESS 3.12)
  file(
    GLOB THRUST_EXAMPLES
    RELATIVE ${PROJECT_SOURCE_DIR}/examples
    ${THRUST_EXAMPLE_GLOBS}
    CONFIGURE_DEPENDS
  )
else ()
  file(
    GLOB THRUST_EXAMPLES
    RELATIVE ${PROJECT_SOURCE_DIR}/examples
    ${THRUST_EXAMPLE_GLOBS}
  )
endif ()

set(THRUST_EXAMPLE_RUN_ARGUMENTS
  -DTHRUST_SOURCE=${CMAKE_SOURCE_DIR}
  -DTHRUST_FILECHECK_ENABLED=${THRUST_EXAMPLE_FILECHECK_ENABLED}
  -DTHRUST_FILECHECK=${THRUST_EXAMPLE_FILECHECK_PATH}
  -P "${CMAKE_SOURCE_DIR}/cmake/run_example.cmake")

foreach (THRUST_EXAMPLE_SOURCE IN LISTS THRUST_EXAMPLES)
  # TODO: Per-example flags.

  get_filename_component(THRUST_EXAMPLE_CATEGORY ${THRUST_EXAMPLE_SOURCE} DIRECTORY)
  if (NOT ("" STREQUAL "${THRUST_EXAMPLE_CATEGORY}"))
    set(THRUST_EXAMPLE_CATEGORY "${THRUST_EXAMPLE_CATEGORY}.")
  endif ()

  get_filename_component(THRUST_EXAMPLE_NAME ${THRUST_EXAMPLE_SOURCE} NAME_WE)

  set(THRUST_EXAMPLE "thrust.example.${THRUST_EXAMPLE_CATEGORY}${THRUST_EXAMPLE_NAME}")

  if (NOT "CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
    # Example files are generally .cu; if CUDA is not enabled, CMake doesn't know what to
    # do with them. But since they are pretty much just C++, we can compile them with
    # non-nvcc C++ compilers... but we need to tell CMake that they are, in fact, just C++.
    set_source_files_properties(${PROJECT_SOURCE_DIR}/examples/${THRUST_EXAMPLE_SOURCE}
      PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "${THRUST_TREAT_FILE_AS_CXX}")
  endif ()

  add_executable(
    ${THRUST_EXAMPLE}
    ${PROJECT_SOURCE_DIR}/examples/${THRUST_EXAMPLE_SOURCE}
  )

  target_include_directories(
    ${THRUST_EXAMPLE}
    PUBLIC ${PROJECT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/examples
  )

  target_link_libraries(${THRUST_EXAMPLE}
    ${THRUST_ADDITIONAL_LIBRARIES})

  add_test(NAME ${THRUST_EXAMPLE}
    COMMAND ${CMAKE_COMMAND}
      -DTHRUST_EXAMPLE=${THRUST_EXAMPLE}
      -DTHRUST_BINARY=$<TARGET_FILE:${THRUST_EXAMPLE}>
      ${THRUST_EXAMPLE_RUN_ARGUMENTS})

  if ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}" AND THRUST_ENABLE_EXAMPLES_WITH_RDC)
    set(THRUST_EXAMPLE_RDC "thrust.example.${THRUST_EXAMPLE_CATEGORY}rdc.${THRUST_EXAMPLE_NAME}")

    add_executable(
      ${THRUST_EXAMPLE_RDC}
      ${PROJECT_SOURCE_DIR}/examples/${THRUST_EXAMPLE_SOURCE}
    )

    target_include_directories(
      ${THRUST_EXAMPLE_RDC}
      PUBLIC ${PROJECT_SOURCE_DIR}
      PRIVATE ${PROJECT_SOURCE_DIR}/examples
    )

    target_link_libraries(${THRUST_EXAMPLE_RDC}
      ${THRUST_ADDITIONAL_LIBRARIES})

    set_target_properties(${THRUST_EXAMPLE_RDC}
      PROPERTIES CUDA_SEPERABLE_COMPILATION ON)

    add_test(NAME ${THRUST_EXAMPLE_RDC}
      COMMAND ${CMAKE_COMMAND}
        -DTHRUST_EXAMPLE=${THRUST_EXAMPLE}
        -DTHRUST_BINARY=$<TARGET_FILE:${THRUST_EXAMPLE_RDC}>
        ${THRUST_EXAMPLE_RUN_ARGUMENTS})
  endif ()
endforeach ()

