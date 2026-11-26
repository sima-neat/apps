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
  OUTPUT_TARGET_VAR OBB_TARGET
  UNIT_TEST
  E2E_TEST
)

set(OBB_UTILS "${CMAKE_CURRENT_LIST_DIR}/utils/oriented_boxes.cpp")
target_sources(${OBB_TARGET} PRIVATE ${OBB_UTILS})

# The unit test checks the Insight objects in process, so it compiles the utils too.
if (TARGET multi-stream-obb-detector_unit_test)
  target_sources(multi-stream-obb-detector_unit_test PRIVATE ${OBB_UTILS})
endif()
