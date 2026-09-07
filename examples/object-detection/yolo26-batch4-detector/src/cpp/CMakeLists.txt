cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(yolo26-batch4-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "yolo26-batch4-detector"
  UNIT_TEST
  E2E_TEST
  OUTPUT_TARGET_VAR YOLO26_BATCH4_DETECTOR_TARGET
)
