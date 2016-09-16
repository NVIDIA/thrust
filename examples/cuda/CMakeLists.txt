
FILE(GLOB SOURCES_CU  *.cu)
FILE(GLOB SOURCES_CPP *.cpp)
FILE(GLOB SOURCES_H *.h)
set(SOURCES_BACKEND ${SOURCES_CU} ${SOURCES_CPP} ${SOURCES_H})

install(FILES ${SOURCES_BACKEND} DESTINATION "examples/cuda" COMPONENT examples)

if (NOT "x${DEVICE_BACKEND}" STREQUAL "xCUDA")
  return()
endif()

list(LENGTH SOURCES_BACKEND index)
message(STATUS "Found ${index} examples/cuda")

set(targets_backend "")
foreach (src ${SOURCES_BACKEND})
  get_filename_component(exec_name ${src} NAME_WE)
  set(target example-${exec_name})
  thrust_add_executable(${target} ${src})
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${exec_name})
  install(TARGETS ${target} DESTINATION "examples/cuda/${HOST_BACKEND}_host_${DEVICE_BACKEND}_device_${THRUST_MODE}" OPTIONAL COMPONENT examples-bin)
  list(APPEND targets_backend ${target})
endforeach()

set(targets ${targets} ${targets_backend} PARENT_SCOPE)


