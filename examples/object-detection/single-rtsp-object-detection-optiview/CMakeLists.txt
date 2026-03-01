cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(single-rtsp-object-detection-optiview LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/ExampleModule.cmake")
sima_neat_apps_module(
  "single-rtsp-object-detection-optiview"
  OUTPUT_TARGET_VAR YOLOV8_ONE_RTSP_OPTIVIEW_TARGET
  OPTIVIEW_E2E_TEST
)

if (BUILD_TESTING)
  add_test(
    NAME single-rtsp-object-detection-optiview.requires_rtsp
    COMMAND $<TARGET_FILE:${YOLOV8_ONE_RTSP_OPTIVIEW_TARGET}>
  )
  set_tests_properties(single-rtsp-object-detection-optiview.requires_rtsp PROPERTIES
    WILL_FAIL TRUE
    LABELS "unit"
  )
endif()
