cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(single-stream-thermal-face-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "single-stream-thermal-face-detector"
  OUTPUT_TARGET_VAR THERMAL_FACE_INSIGHT_TARGET
  UNIT_TEST
  METADATA_E2E_TEST
)

if (BUILD_TESTING)
  add_test(
    NAME single-stream-thermal-face-detector.requires_rtsp
    COMMAND $<TARGET_FILE:${THERMAL_FACE_INSIGHT_TARGET}> --config /nonexistent/single-stream-thermal-face-detector-config.yaml
  )
  set_tests_properties(single-stream-thermal-face-detector.requires_rtsp PROPERTIES
    WILL_FAIL TRUE
    LABELS "unit"
  )
endif()
