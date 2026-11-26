cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(fastflow-anomaly-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "fastflow-anomaly-detector"
  OUTPUT_TARGET_VAR FASTFLOW_TARGET
  UNIT_TEST
  E2E_TEST
)

set(FASTFLOW_UTILS "${CMAKE_CURRENT_LIST_DIR}/utils/anomaly_regions.cpp")
target_sources(${FASTFLOW_TARGET} PRIVATE ${FASTFLOW_UTILS})

# The unit test checks the region extraction in process, so it compiles the utils too.
if (TARGET fastflow-anomaly-detector_unit_test)
  target_sources(fastflow-anomaly-detector_unit_test PRIVATE ${FASTFLOW_UTILS})
endif()
