cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(pcie-high-density-multi-stream-object-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "pcie-high-density-multi-stream-object-detector"
  OUTPUT_TARGET_VAR PCIE_HIGH_DENSITY_TARGET
)

target_sources("${PCIE_HIGH_DENSITY_TARGET}"
  PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/stream_demux.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../common/app_config.cpp"
)
