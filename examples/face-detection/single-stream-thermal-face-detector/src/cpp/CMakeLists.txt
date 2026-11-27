cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(single-stream-thermal-face-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module("single-stream-thermal-face-detector" UNIT_TEST METADATA_E2E_TEST
                      OUTPUT_TARGET_VAR THERMAL_FACE_TARGET)

# The face decode and the pose message; the unit test includes the same header.
target_sources(${THERMAL_FACE_TARGET} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/utils/yolov5_face.h")
