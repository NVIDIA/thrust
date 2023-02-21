# Parse version information from version.h in source tree
set(_THRUST_VERSION_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
if(EXISTS "${_THRUST_VERSION_INCLUDE_DIR}/thrust/version.h")
  set(_THRUST_VERSION_INCLUDE_DIR "${_THRUST_VERSION_INCLUDE_DIR}" CACHE FILEPATH "" FORCE) # Clear old result
  set_property(CACHE _THRUST_VERSION_INCLUDE_DIR PROPERTY TYPE INTERNAL)
endif()
