# This file must be included before enabling any languages.

if (NOT ("${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "" OR
    "${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "${CMAKE_CXX_COMPILER}"))
  set(tmp "${CMAKE_CUDA_HOST_COMPILER}")
  unset(CMAKE_CUDA_HOST_COMPILER CACHE)
  message(FATAL_ERROR
    "For convenience, Thrust's test harness uses CMAKE_CXX_COMPILER for the "
    "CUDA host compiler. Refusing to overwrite specified "
    "CMAKE_CUDA_HOST_COMPILER -- please reconfigure without setting this "
    "variable. Currently:\n"
    "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}\n"
    "CMAKE_CUDA_HOST_COMPILER=${tmp}"
  )
endif ()
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
