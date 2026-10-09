cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(single-stream-instance-segmenter LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")
sima_neat_apps_module("single-stream-instance-segmenter"
  OUTPUT_TARGET_VAR SEGMENTER_TARGET UNIT_TEST E2E_TEST)

# The unit test compiles the application source so it can exercise the decode boundary
# and the shared pull-outcome helper directly, which needs the same libraries and
# definitions as the application target.
if (TARGET single-stream-instance-segmenter_unit_test)
  target_link_libraries(single-stream-instance-segmenter_unit_test
    PRIVATE SimaNeatApps::support_runtime)
  target_compile_definitions(single-stream-instance-segmenter_unit_test PRIVATE
    $<TARGET_PROPERTY:${SEGMENTER_TARGET},COMPILE_DEFINITIONS>)
endif()
