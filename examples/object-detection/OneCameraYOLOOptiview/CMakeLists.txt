cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(OneCameraYOLOOptiview LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/ExampleModule.cmake")
sima_neat_apps_module(
  "OneCameraYOLOOptiview"
  OUTPUT_TARGET_VAR ONECAM_YOLO_OPTIVIEW_TARGET
  OPTIVIEW_E2E_TEST
)

if (BUILD_TESTING)
  add_test(
    NAME OneCameraYOLOOptiview.requires_rtsp
    COMMAND $<TARGET_FILE:${ONECAM_YOLO_OPTIVIEW_TARGET}>
  )
  set_tests_properties(OneCameraYOLOOptiview.requires_rtsp PROPERTIES
    WILL_FAIL TRUE
    PASS_REGULAR_EXPRESSION "\\[ERR\\] Missing --rtsp <url>"
  )
endif()
