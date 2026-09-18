cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(yolov8-instance-segmenter LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")
sima_neat_apps_module("yolov8-instance-segmenter" OUTPUT_TARGET_VAR SEGMENTER_TARGET UNIT_TEST E2E_TEST)

if (BUILD_TESTING)
  target_link_libraries(yolov8-instance-segmenter_unit_test PRIVATE SimaNeatApps::support_runtime)
  target_compile_definitions(yolov8-instance-segmenter_unit_test PRIVATE
    $<TARGET_PROPERTY:${SEGMENTER_TARGET},COMPILE_DEFINITIONS>)
endif()
