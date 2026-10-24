cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(multi-stream-blazepose3d LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "multi-stream-blazepose3d"
  UNIT_TEST
  E2E_TEST
  OUTPUT_TARGET_VAR MULTISTREAM_BLAZEPOSE3D_TARGET
)

if (BUILD_TESTING)
  find_package(OpenCV REQUIRED COMPONENTS core videoio)
  target_link_libraries(multi-stream-blazepose3d_e2e_test PRIVATE opencv_videoio)
endif()
