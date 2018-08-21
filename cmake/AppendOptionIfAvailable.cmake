include_guard(GLOBAL)
include(CheckCXXCompilerFlag)
include(CheckCUDACompilerFlag)

set(_COUNTER 0 CACHE STRING "Counter for `append_option_if_available`")

macro (APPEND_OPTION_IF_AVAILABLE _LANGUAGE _FLAG _LIST)
set(_AVAILABLE_UNIQUE _AVAILABLE_${_COUNTER})

if     ("CXX"  STREQUAL "${_LANGUAGE}")
  check_cxx_compiler_flag(${_FLAG} ${_AVAILABLE_UNIQUE} "${_FLAG}")
elseif ("CUDA" STREQUAL "${_LANGUAGE}")
  check_cuda_compiler_flag(${_FLAG} ${_AVAILABLE_UNIQUE} "${_FLAG}")
else ()
  message(FATAL_ERROR "Language ${_LANGUAGE} is not supported!")
endif ()

if (${_AVAILABLE_UNIQUE})
  list(APPEND ${_LIST} ${_FLAG})
endif ()

math(EXPR _COUNTER "${_COUNTER} + 1")
endmacro ()

