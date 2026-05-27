cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(human-pose-estimator LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/ExampleModule.cmake")
sima_neat_apps_module(
  "human-pose-estimator"
  OUTPUT_TARGET_VAR
    pose_estimation_example_target
  UNIT_TEST
  E2E_TEST
)

target_sources(${pose_estimation_example_target}
  PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/utils/config.cpp"
)

target_include_directories(${pose_estimation_example_target}
  PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/utils"
)

target_compile_definitions(${pose_estimation_example_target}
  PRIVATE
    SIMPLE_POSE_ESTIMATION_OVERLAY_PIPELINE_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}/.."
)
