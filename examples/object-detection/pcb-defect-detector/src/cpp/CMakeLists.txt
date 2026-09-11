cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(pcb-defect-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")
sima_neat_apps_module("pcb-defect-detector" UNIT_TEST E2E_TEST)

# This example's E2E test rescales images, so it needs an image codec that the
# shared test helper does not link.
if (TARGET pcb-defect-detector_e2e_test)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(PCB_E2E_OPENCV REQUIRED IMPORTED_TARGET opencv4)
  target_link_libraries(pcb-defect-detector_e2e_test PRIVATE PkgConfig::PCB_E2E_OPENCV)
endif()
