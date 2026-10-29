cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(multi-stream-obb-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "multi-stream-obb-detector"
  UNIT_TEST
  E2E_TEST
  OUTPUT_TARGET_VAR MULTISTREAM_OBB_DETECTOR_TARGET
)

# The unit test compiles the application's helpers, so it needs the application's build setup.
if (TARGET multi-stream-obb-detector_unit_test)
  target_compile_definitions(multi-stream-obb-detector_unit_test
    PRIVATE
      $<TARGET_PROPERTY:${MULTISTREAM_OBB_DETECTOR_TARGET},COMPILE_DEFINITIONS>
  )
  target_link_libraries(multi-stream-obb-detector_unit_test
    PRIVATE
      SimaNeatApps::support_runtime
  )
endif()
