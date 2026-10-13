cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(single-stream-instance-segmenter LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")
sima_neat_apps_module("single-stream-instance-segmenter" UNIT_TEST E2E_TEST)

# The unit test exercises the shared pull-outcome helper, which needs the Neat headers.
if (TARGET single-stream-instance-segmenter_unit_test)
  target_link_libraries(single-stream-instance-segmenter_unit_test
    PRIVATE
      SimaNeatApps::support_runtime
  )
endif()
