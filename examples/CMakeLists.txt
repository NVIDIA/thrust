# message(STATUS "Adding \"examples\"")

#aux_source_directory("testing" sources)
FILE(GLOB SOURCES_CU  *.cu)
FILE(GLOB SOURCES_CPP *.cpp)
set(SOURCES ${SOURCES_CU})

list(LENGTH SOURCES index)
message(STATUS "Found ${index} examples")

set(targets "")
foreach (src ${SOURCES})
  get_filename_component(exec_name ${src} NAME_WE)
  set(target example-${exec_name})
  thrust_add_executable(${target} ${src})
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${exec_name})
  install(TARGETS ${target} DESTINATION "examples/${HOST_BACKEND}_host_${DEVICE_BACKEND}_device_${THRUST_MODE}" OPTIONAL COMPONENT examples-bin)
  list(APPEND targets ${target})
endforeach()

add_subdirectory(cuda)
add_subdirectory(omp)
add_subdirectory(cpp_integration)

add_custom_target(examples-bin DEPENDS ${targets})
add_custom_target(install-examples-bin
  COMMAND 
      "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=examples-bin
      -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
)

install(FILES ${SOURCES} DESTINATION "examples" COMPONENT examples)

