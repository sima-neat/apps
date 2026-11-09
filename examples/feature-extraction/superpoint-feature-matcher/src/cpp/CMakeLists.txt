cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(superpoint-feature-matcher LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")
sima_neat_apps_module("superpoint-feature-matcher" UNIT_TEST E2E_TEST
  OUTPUT_TARGET_VAR matcher_target)

# The inlier ratio fits a fundamental matrix, and the stages and render workers run on their
# own threads.
find_package(Threads REQUIRED)
target_link_libraries(${matcher_target} PRIVATE opencv_calib3d Threads::Threads)
