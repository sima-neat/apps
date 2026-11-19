cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(edgetam-video-object-segmenter LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "edgetam-video-object-segmenter"
  UNIT_TEST
  E2E_TEST
  OUTPUT_TARGET_VAR EDGETAM_VIDEO_OBJECT_SEGMENTER_TARGET
)

# The tracking steps are compiled into the application and into the unit test, which tests them
# directly.
target_sources(
  "${EDGETAM_VIDEO_OBJECT_SEGMENTER_TARGET}"
  PRIVATE "${CMAKE_CURRENT_LIST_DIR}/utils/tracker.cpp"
)
if (TARGET edgetam-video-object-segmenter_unit_test)
  target_sources(
    edgetam-video-object-segmenter_unit_test
    PRIVATE "${CMAKE_CURRENT_LIST_DIR}/utils/tracker.cpp"
  )
endif()
