cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(single-stream-object-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "single-stream-object-detector"
  OUTPUT_TARGET_VAR YOLO26_ONE_RTSP_INSIGHT_TARGET
  UNIT_TEST
  METADATA_E2E_TEST
)

if (BUILD_TESTING)
  add_test(
    NAME single-stream-object-detector.requires_rtsp
    COMMAND $<TARGET_FILE:${YOLO26_ONE_RTSP_INSIGHT_TARGET}> --config /nonexistent/single-rtsp-config.yaml
  )
  set_tests_properties(single-stream-object-detector.requires_rtsp PROPERTIES
    WILL_FAIL TRUE
    LABELS "unit"
  )
endif()
