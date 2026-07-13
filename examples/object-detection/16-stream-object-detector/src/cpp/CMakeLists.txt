cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(16-stream-object-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "16-stream-object-detector"
  UNIT_TEST
  OUTPUT_TARGET_VAR SIXTEEN_STREAM_OBJECT_DETECTION_INSIGHT_TARGET
)

# Do not embed the cross-build Core directory in a binary handed to a DevKit.
# The matching Core/Internals packages must be installed in their normal target
# locations; the dynamic loader then resolves them without a developer path.
set_target_properties(
  ${SIXTEEN_STREAM_OBJECT_DETECTION_INSIGHT_TARGET}
  PROPERTIES SKIP_BUILD_RPATH TRUE
)
