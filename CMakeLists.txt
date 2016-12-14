cmake_minimum_required(VERSION 3.0)
project(Thrust CXX)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)

file(READ "thrust/version.h" thrust_version_file)
string(REGEX MATCH "THRUST_VERSION ([0-9]+)" DUMMY ${thrust_version_file})
set(thrust_version ${CMAKE_MATCH_1})
#message("thrust_version= ${thrust_version}")
math(EXPR Thrust_VERSION_MAJOR "(${thrust_version} / 100000)")
math(EXPR Thrust_VERSION_MINOR "(${thrust_version} / 100) % 1000")
math(EXPR Thrust_VERSION_PATCH " ${thrust_version} % 100")

message(STATUS "Thrust version ${Thrust_VERSION_MAJOR}.${Thrust_VERSION_MINOR}.${Thrust_VERSION_PATCH}")


include(CTest)
enable_testing()

function(print_flags flags)
  message("${flags}:")
  set(flags ${${flags}})
  set(__is_name True)
  foreach(arg ${flags})
    if (__is_name)
      set(__arg_name ${arg})
      set(__is_name False)
    else()
      separate_arguments(arg)
      set(arg ${arg})
      message(" | ${__arg_name} : '${arg}'")
      set(__is_name True)
    endif()
  endforeach()
endfunction()


set(
  GNU_COMPILER_FLAGS
  WARN_ALL             "-Wall"
  WARNINGS_AS_ERRORS   "-Werror"
  RELEASE              "-O2"
  DEBUG                "-g"
  EXCEPTION_HANDLING   " "
  CPP                  " "
  OMP                  "-fopenmp"
  TBB                  " "
  CUDA                 " "
  CUDA_BULK          " "
  WORKAROUNDS          " "
  C++03                " "
  C++11                "-std=c++11"
  )
set(
  GNU_LINKER_FLAGS
  DEBUG " "
  RELEASE " "
  WORKAROUNDS " "
  CPP " "
  OMP "-fopenmp"
  TBB " "
  CUDA " "
  CUDA_BULK " "
  )

set(
  CLANG_COMPILER_FLAGS
  WARN_ALL             "-Wall"
  WARNINGS_AS_ERRORS   "-Werror"
  RELEASE              "-O2"
  DEBUG                "-g"
  EXCEPTION_HANDLING   " "
  CPP                  " "
  OMP                  "-fopenmp"
  TBB                  " "
  CUDA                 " "
  CUDA_BULK            " "
  WORKAROUNDS          " "
  C++03                " "
  C++11                "-std=c++11"
  )
set(
  CLANG_LINKER_FLAGS
  DEBUG " "
  RELEASE " "
  WORKAROUNDS " " #-stdlib=libstdc++"
  CPP " "
  OMP "-fopenmp"
  TBB " "
  CUDA " "
  CUDA_BULK " "
  )

set(
  MSVC_COMPILER_FLAGS
  WARN_ALL             "/Wall"
  WARNINGS_AS_ERRORS   "/Wx"
  RELEASE              "/Ox"
  DEBUG                "/Zi -D_DEBUG /MTd"
  EXCEPTION_HANDLING   "/EHsc"
  CPP                  " "
  OMP                  "/openmp"
  TBB                  " "
  CUDA                 " "
  CUDA_BULK            " "
  WORKAROUNDS          "/DNOMINMAX /wd4503"
  C++03                " "
  C++11                "-std=c++11"
  )
set(
  MSVC_LINKER_FLAGS
  DEBUG                "/debug"
  RELEASE              " "
  WORKAROUND           "/nologo"
  CPP                  " "
  OMP                  "/openmp"
  TBB                  " "
  CUDA                 " "
  CUDA_BULK            " "
  WORKAROUNDS          " "
  )

set(NV_LINKER_FLAGS ${GNU_LINKER_FLAGS})

print_flags(MSVC_COMPILER_FLAGS)


function(add_option OPTION_NAME DESCRIPTION TYPE)
  if (${ARGC} EQUAL 3)
    message(FATAL_ERROR "No option value [list] is provided")
  endif()
  if (${OPTION_NAME} AND "x${TYPE}" STREQUAL "xSTRING")
    LIST(FIND ARGN ${${OPTION_NAME}} index)
    if (index EQUAL -1)
      message(FATAL_ERROR "Invalid value '${${OPTION_NAME}}' for '${DESCRIPTION}'")
    endif()
  endif()
  set(value_list ${ARGN})
  LIST(GET value_list  0 default_value)
  LIST(SORT value_list)
  set(${OPTION_NAME} ${default_value} CACHE ${TYPE} ${DESCRIPTION})
  if ("x${TYPE}" STREQUAL "xSTRING")
    set_property(CACHE ${OPTION_NAME} PROPERTY STRINGS ${value_list})
  endif()
endfunction()

add_option(CUDA_ARCH  "Compute capability code generation" STRING sm_61
  sm_30 sm_32 sm_35 sm_37
  sm_50 sm_52 sm_61)
add_option(HOST_BACKEND   "The host   backend to target" STRING CPP OMP TBB)
add_option(DEVICE_BACKEND "The device backend to target" STRING CUDA CUDA_BULK CPP OMP TBB)
add_option(CUDA_CDP "Enable CUDA dynamic parallelism" BOOL False)
add_option(CXX_STD "C++ standard" STRING C++03 C++11)
add_option(THRUST_MODE "Release versus debug mode" STRING RELEASE DEBUG)

if (WIN32)
  set(WINNT True)
  set(NOT_WINNT False)
  add_option(MSVC_VERSION "MS Visual C++ version" STRING NONE 8.0 9.0 10.0 11.0 12.0 13.0 1900)
else()
  set(WINNT False)
  set(NOT_WINNT True)
endif()
add_option(WARN_ALL "Enable all compilation warnings" BOOL ${NOT_WINNT})
add_option(WARN_ERROR "Treat warnings as errors" BOOL ${NOT_WINNT})

IF(NOT CMAKE_BUILD_TYPE)
  # possible cmake bug (?) : RelWithDebInfo passes -DNDEBUG
    SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Helpers
macro(set_thrust_flags THRUST_FLAGS_)
  set(${THRUST_FLAGS_} "-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_${HOST_BACKEND}")
  LIST(APPEND ${THRUST_FLAGS_} "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_${DEVICE_BACKEND}")

  if (THRUST_MODE STREQUAL "DEBUG")
    LIST(APPEND ${THRUST_FLAGS_} "-DTHRUST_DEBUG")
  endif()
endmacro()

macro(get_compiler_id COMPILER_ID_)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(${COMPILER_ID_} "GNU")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(${COMPILER_ID_} "CLANG")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(${COMPILER_ID_} "CLANG")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(${COMPILER_ID_} "Intel")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(${COMPILER_ID_} "MSVC")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    set(${COMPILER_ID_} "PGI")
  endif()
endmacro()

macro(find_key_value LIST_ KEY_ VALUE_)
  LIST(FIND ${LIST_} ${KEY_}  index_)
  if (index_ EQUAL -1) 
    message(FATAL_ERROR "${KEY_} is not found in ${LIST_}." )
  endif()
  math(EXPR index_ "${index_}+1")
  LIST(GET ${LIST_} ${index_} ${VALUE_})
  separate_arguments(${VALUE_})
endmacro()

macro(set_cc_compiler_flags CC_COMPILER_FLAGS_)
  get_compiler_id(CXX_)
  set(CXX_ ${CXX_}_COMPILER_FLAGS)

  find_key_value(${CXX_} EXCEPTION_HANDLING flags_)
  LIST(APPEND ${CC_COMPILER_FLAGS_} ${flags_})

  find_key_value(${CXX_} ${HOST_BACKEND} flags_)
  LIST(APPEND ${CC_COMPILER_FLAGS_} ${flags_})
  
  find_key_value(${CXX_} ${DEVICE_BACKEND} flags_)
  LIST(APPEND ${CC_COMPILER_FLAGS_} ${flags_})

  if (${WARN_ALL})
    find_key_value(${CXX_} WARN_ALL flags_)
    LIST(APPEND ${CC_COMPILER_FLAGS_} ${flags_})
  endif()
  
  if (${WARN_ERROR})
    find_key_value(${CXX_} WARNINGS_AS_ERRORS flags_)
    LIST(APPEND ${CC_COMPILER_FLAGS_} ${flags_})
  endif()

  find_key_value(${CXX_} ${CXX_STD} flags_)
  LIST(APPEND ${CC_COMPILER_FLAGS_} ${flags_})
endmacro()

macro(set_nv_compiler_flags NV_COMPILER_FLAGS_)
  set(MACHINE_ARCH_ ${CUDA_ARCH})
  # Transform sm_XX to compute_XX
  string(REGEX REPLACE "sm" "compute"  VIRTUAL_ARCH_ ${MACHINE_ARCH_})
  # Produce -gencode flags like this: -gencode=arch=compute_XX,code=\"sm_XX,compute_XX\"
  LIST(APPEND ${NV_COMPILER_FLAGS_} "-gencode=arch=${VIRTUAL_ARCH_},\\\"code=${MACHINE_ARCH_},${VIRTUAL_ARCH_}\\\"")

  if ("${THRUST_MODE}" STREQUAL "DEBUG")
    # turn on debug mode
    # XXX make this work when we've debugged nvcc -G
#    LIST(APPEND ${NV_COMPILER_FLAGS_} "-G")    
  endif()

  if ((NOT "${DEVICE_BACKEND}" STREQUAL "CUDA") AND (NOT "${DEVICE_BACKEND}"  STREQUAL "CUDA_BULK"))
    LIST(APPEND ${NV_COMPILER_FLAGS_} "--x=c++")
  endif()

  if (${CUDA_CDP})
#    LIST(APPEND ${NV_COMPILER_FLAGS_} "-rdc=true")
  endif()

  # Untested on OSX 10.8.*
  if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    if ("${CMAKE_SYSTEM_VERSION}" STREQUAL "10.8.")
      LIST(APPEND ${NV_COMPILER_FLAGS_} "-ccbin ${CMAKE_CXX_COMPILER}")
    endif()
  endif()
endmacro()

macro(set_linker_flags LINKER_FLAGS_)
  get_compiler_id(LINK_)
  set(LINK_ ${LINK_}_LINKER_FLAGS)

  find_key_value(${LINK_} ${THRUST_MODE} flags_)
  LIST(APPEND ${LINKER_FLAGS_} ${flags_})

  find_key_value(${LINK_} WORKAROUNDS flags_)
  LIST(APPEND ${LINKER_FLAGS_} ${flags_})
  
  find_key_value(${LINK_} ${HOST_BACKEND} flags_)
  LIST(APPEND ${LINKER_FLAGS_} ${flags_})
  
  find_key_value(${LINK_} ${DEVICE_BACKEND} flags_)
  LIST(APPEND ${LINKER_FLAGS_} ${flags_})
endmacro()

macro(thrust_add_executable TARGET)
  if ((NOT "${DEVICE_BACKEND}" STREQUAL "CUDA") AND (NOT "${DEVICE_BACKEND}" STREQUAL "CUDA_BULK")) # AND "${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CXX)
    add_executable(${TARGET} ${ARGN})
    set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(${TARGET} PROPERTIES COMPILE_FLAGS "-x c++")
  else()
    cuda_add_executable(${TARGET} ${ARGN})
  endif()
endmacro()

#macro(thrust_include_directories TARGET)
#  if (NOT "${DEVICE_BACKEND}" STREQUAL "CUDA") # AND "${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
#    target_include_directories(${TARGET} PRIVATE ${ARGN})
#  else()
#    cuda_include_directories(${ARGN})
#  endif()
#endmacro()

# Find backends

find_package(CUDA)
find_package(OpenMP)

# Set flags

set_thrust_flags(THRUST_FLAGS)
set_cc_compiler_flags(CC_FLAGS)
set_nv_compiler_flags(NV_FLAGS)
set_linker_flags(LINKER_FLAGS)

# Debug output
# message("THRUST_FLAGS= ${THRUST_FLAGS}")
# message("CC_FLAGS= ${CC_FLAGS}")
# message("NV_FLAGS= ${NV_FLAGS}")
# message("LINKER_FLAGS= ${LINKER_FLAGS}")

string (REPLACE ";" " " CC_FLAGS_STR "${CC_FLAGS} ${THRUST_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CC_FLAGS_STR}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NV_FLAGS})
string (REPLACE ";" " " LINKER_FLAGS_STR "${LINKER_FLAGS}")
set(CMAKE_EXEC_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LINKER_FLAGS_STR}")

# Enable separable compilation when building with CUDA Dynamic Parallelism
set(CUDA_SEPARABLE_COMPILATION ${CUDA_CDP})
# and find "cudadevrt" library for linking, otherwise <<<,>>> will fail to build
if (${CUDA_CDP})
  cuda_find_library_local_first(CUDADEVRT_LIBRARY cudadevrt "\"cudadevrt\" library")
  if ("${CUDADEVRT_LIBRARY}" STREQUAL "CUDADEVRT_LIBRARY-NOTFOUND")
    message(FATAL_ERROR "\"cudadevrt\" library not found. Consider disabling CUDA_CDP.")
  endif()
  link_libraries(${CUDADEVRT_LIBRARY})
endif()


include_directories(${CMAKE_SOURCE_DIR})
cuda_include_directories(${CMAKE_SOURCE_DIR})

# Add targets

# thrust target
install(DIRECTORY ${CMAKE_SOURCE_DIR}/thrust/ DESTINATION thrust COMPONENT thrust)
install(FILES ${CMAKE_SOURCE_DIR}/CHANGELOG DESTINATION thrust COMPONENT thrust)
add_custom_target(install-thrust
  COMMAND
      "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=thrust
      -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
)

# add examples, testing and performance testing targets
add_subdirectory(examples)
add_subdirectory(testing)
add_subdirectory(performance)

### make zip acrhive

set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_VERSION "${Thrust_VERSION_MAJOR}.${Thrust_VERSION_MINOR}.${Thrust_VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION_MAJOR "${Thrust_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${Thrust_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${Thrust_VERSION_PATCH}")
set(CPACK_COMPONENTS_ALL thrust examples)
set(CPACK_ZIP_USE_DISPLAY_NAME_IN_FILENAME ON)
set(CPACK_PACKAGE_FILE_NAME "Thrust-${CPACK_PACKAGE_VERSION}")
include(CPack)
cpack_add_component(thrust DISPLAY_NAME "headers")
cpack_add_component(examples DISPLAY_NAME "examples")
