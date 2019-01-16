include_guard(GLOBAL)
include(CheckCXXCompilerFlag)

macro (APPEND_OPTION_IF_AVAILABLE _FLAG _LIST)

set(_VAR "CXX_FLAG_${_FLAG}")
check_cxx_compiler_flag(${_FLAG} ${_VAR})

if (${${_VAR}})
  list(APPEND ${_LIST} ${_FLAG})
endif ()

endmacro ()

