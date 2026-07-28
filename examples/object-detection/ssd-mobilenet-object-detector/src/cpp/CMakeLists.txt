cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(ssd-mobilenet-object-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")
sima_neat_apps_module("ssd-mobilenet-object-detector" UNIT_TEST E2E_TEST
                      OUTPUT_TARGET_VAR ssd_mobilenet_target)

# Absolute source dir so the app can locate its packaged labels regardless of cwd/build layout.
target_compile_definitions("${ssd_mobilenet_target}"
  PRIVATE SIMANEAT_APPS_EXAMPLE_SOURCE_ABS_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
