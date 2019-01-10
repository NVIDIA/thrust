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

if ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  enable_language(CUDA)
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

  set(THRUST_TREAT_FILE_AS_CXX "/TP")
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

  set(THRUST_TREAT_FILE_AS_CXX "-x c++")
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


if ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  append_option_if_available(CUDA "-rdc=true" THRUST_OPTIONS_RDC)
endif ()

set(THRUST_OPTIONS_DEBUG ${THRUST_OPTIONS_WARNINGS})
set(THRUST_OPTIONS_RELEASE ${THRUST_OPTIONS_WARNINGS})

include(CTest)
enable_testing()

# Handle tests

list(APPEND THRUST_TESTFRAMEWORK_FILES testing/unittest/testframework.cu)
if ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  list(APPEND THRUST_TESTFRAMEWORK_FILES testing/unittest/cuda/testframework.cu)
else ()
  # when CUDA is disabled, explain to CMake that testframework.cu is actually a C++ file
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

# list of tests that aren't implemented for all backends, but are implemented for CUDA
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

# list of tests that aren't implemented for all backends, but are implemented for CPP
set(THRUST_PARTIALLY_IMPLEMENTED_CPP
)

# list of tests that aren't implemented for all backends, but are implemented for TBB
set(THRUST_PARTIALLY_IMPLEMENTED_TBB
)

# list of tests that aren't implemented for all backends, but are implemented for OMP
set(THRUST_PARTIALLY_IMPLEMENTED_OMP
)

# list of all partially implemented tests
set(THRUST_PARTIALLY_IMPLEMENTED
  ${THRUST_PARTIALLY_IMPLEMENTED_CUDA}
  ${THRUST_PARTIALLY_IMPLEMENTED_CPP}
  ${THRUST_PARTIALLY_IMPLEMENTED_TBB}
  ${THRUST_PARTIALLY_IMPLEMENTED_OMP}
)

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
    # this test is partially implemented on _some_ backends
    if (NOT "${THRUST_TEST_NAME}" IN_LIST THRUST_PARTIALLY_IMPLEMENTED_${THRUST_DEVICE_SYSTEM})
      # but not on the selected one
      set(THRUST_TEST_CREATION_ADDITIONAL EXCLUDE_FROM_ALL)
      set(THRUST_TEST_ADD_TO_CTEST OFF)
    endif ()
  endif ()

  set(THRUST_TEST "thrust.test.${THRUST_TEST_CATEGORY}${THRUST_TEST_NAME}")

  if (NOT "CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
    # test files are generally .cu; if CUDA is not enabled, CMake doesn't know what to
    # do with them. but since they are pretty much just C++, we can compile them with
    # non-nvcc C++ compilers... but we need to tell CMake that they are, in fact, just C++.
    set_source_files_properties(${PROJECT_SOURCE_DIR}/testing/${THRUST_TEST_SOURCE}
      PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "${THRUST_TREAT_FILE_AS_CXX}")
  endif ()

  add_executable(
    ${THRUST_TEST}
    ${THRUST_TEST_CREATION_ADDITIONAL}

    ${PROJECT_SOURCE_DIR}/testing/${THRUST_TEST_SOURCE}
  )

  target_compile_options(${THRUST_TEST}
    PRIVATE "$<$<CONFIG:DEBUG>:${THRUST_OPTIONS_DEBUG}>"
            "$<$<CONFIG:RELEASE>:${THRUST_OPTIONS_RELEASE}>")

  target_include_directories(
    ${THRUST_TEST}
    PUBLIC ${PROJECT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/testing
  )

  target_link_libraries(${THRUST_TEST}
    thrust_testframework
    ${THRUST_ADDITIONAL_LIBRARIES})

  if (THRUST_TEST_ADD_TO_CTEST)
    add_test(${THRUST_TEST}     ${THRUST_TEST})
  endif ()

  if ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
    set(THRUST_TEST_RDC "thrust.test.${THRUST_TEST_CATEGORY}rdc.${THRUST_TEST_NAME}")

    add_executable(
      ${THRUST_TEST_RDC}
      ${THRUST_TEST_CREATION_ADDITIONAL}

      ${PROJECT_SOURCE_DIR}/testing/${THRUST_TEST_SOURCE}
    )

    target_compile_options(${THRUST_TEST_RDC}
      PRIVATE "$<$<CONFIG:DEBUG>:${THRUST_OPTIONS_DEBUG} ${THRUST_OPTIONS_RDC}>"
              "$<$<CONFIG:RELEASE>:${THRUST_OPTIONS_RELEASE} ${THRUST_OPTIONS_RDC}>")

    target_include_directories(
      ${THRUST_TEST_RDC}
      PUBLIC ${PROJECT_SOURCE_DIR}
      PRIVATE ${PROJECT_SOURCE_DIR}/testing
    )

    target_link_libraries(${THRUST_TEST_RDC}
      thrust_testframework
      ${THRUST_ADDITIONAL_LIBRARIES})

    if (THRUST_TEST_ADD_TO_CTEST)
      add_test(${THRUST_TEST_RDC} ${THRUST_TEST_RDC})
    endif ()
  endif ()
endforeach ()

# Handle examples

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

foreach (THRUST_EXAMPLE_SOURCE IN LISTS THRUST_EXAMPLES)
  # TODO: Per-example flags.

  get_filename_component(THRUST_EXAMPLE_CATEGORY ${THRUST_EXAMPLE_SOURCE} DIRECTORY)
  if (NOT ("" STREQUAL "${THRUST_EXAMPLE_CATEGORY}"))
    set(THRUST_EXAMPLE_CATEGORY "${THRUST_EXAMPLE_CATEGORY}.")
  endif ()

  get_filename_component(THRUST_EXAMPLE_NAME ${THRUST_EXAMPLE_SOURCE} NAME_WE)

  set(THRUST_EXAMPLE "thrust.example.${THRUST_EXAMPLE_CATEGORY}${THRUST_EXAMPLE_NAME}")

  if (NOT "CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
    # example files are generally .cu; if CUDA is not enabled, CMake doesn't know what to
    # do with them. but since they are pretty much just C++, we can compile them with
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

  target_compile_options(${THRUST_EXAMPLE}
    PRIVATE "$<$<CONFIG:DEBUG>:${THRUST_OPTIONS_DEBUG}>"
            "$<$<CONFIG:RELEASE>:${THRUST_OPTIONS_RELEASE}>")

  target_include_directories(
    ${THRUST_EXAMPLE}
    PUBLIC ${PROJECT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/examples
  )

  target_link_libraries(${THRUST_EXAMPLE}
    ${THRUST_ADDITIONAL_LIBRARIES})

  add_test(${THRUST_EXAMPLE}     ${THRUST_EXAMPLE})

  if ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
    set(THRUST_EXAMPLE_RDC "thrust.example.${THRUST_EXAMPLE_CATEGORY}rdc.${THRUST_EXAMPLE_NAME}")

    add_executable(
      ${THRUST_EXAMPLE_RDC}
      ${PROJECT_SOURCE_DIR}/examples/${THRUST_EXAMPLE_SOURCE}
    )

    target_compile_options(${THRUST_EXAMPLE_RDC}
      PRIVATE "$<$<CONFIG:DEBUG>:${THRUST_OPTIONS_DEBUG} ${THRUST_OPTIONS_RDC}>"
              "$<$<CONFIG:RELEASE>:${THRUST_OPTIONS_RELEASE} ${THRUST_OPTIONS_RDC}>")

    target_include_directories(
      ${THRUST_EXAMPLE_RDC}
      PUBLIC ${PROJECT_SOURCE_DIR}
      PRIVATE ${PROJECT_SOURCE_DIR}/examples
    )

    target_link_libraries(${THRUST_EXAMPLE_RDC}
      ${THRUST_ADDITIONAL_LIBRARIES})

    add_test(${THRUST_EXAMPLE_RDC} ${THRUST_EXAMPLE_RDC})
  endif ()
endforeach ()

