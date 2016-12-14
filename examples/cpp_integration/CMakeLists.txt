FILE(GLOB SOURCES_CU  *.cu)
FILE(GLOB SOURCES_CPP *.cpp)
FILE(GLOB SOURCES_H *.h)
set(SOURCES_BACKEND ${SOURCES_CU} ${SOURCES_CPP} ${SOURCES_H})
list(APPEND SOURCES_BACKEND "README")

install(FILES ${SOURCES_BACKEND} DESTINATION "examples/cpp_integration" COMPONENT examples)

if (NOT "x${DEVICE_BACKEND}" STREQUAL "xCUDA")
  return()
endif()

list(LENGTH SOURCES_BACKEND index)
message(STATUS "Found ${index} examples/cpp_integration")

set(targets_backend "")
set(exec_name "cpp_integration")
set(target example-${exec_name})
thrust_add_executable(${target} ${SOURCES_BACKEND})
set_target_properties(${target} PROPERTIES OUTPUT_NAME ${exec_name})
install(TARGETS ${target} DESTINATION "examples/cpp_integration/${HOST_BACKEND}_host_${DEVICE_BACKEND}_device_${THRUST_MODE}" OPTIONAL COMPONENT examples-bin)
list(APPEND targets_backend ${target})

set(targets ${targets} ${targets_backend} PARENT_SCOPE)