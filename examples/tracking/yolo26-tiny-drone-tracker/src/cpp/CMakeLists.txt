cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(yolo26-tiny-drone-tracker LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "yolo26-tiny-drone-tracker"
  OUTPUT_TARGET_VAR DRONE_TRACKER_TARGET
  UNIT_TEST
  E2E_TEST
)

set(DRONE_TRACKER_UTILS "${CMAKE_CURRENT_LIST_DIR}/utils/tracker.cpp")
target_sources(${DRONE_TRACKER_TARGET} PRIVATE ${DRONE_TRACKER_UTILS})

# The unit test checks the tracker in process, so it compiles the utils too, and links the Neat
# Library whose Box the tracker reads.
if (TARGET yolo26-tiny-drone-tracker_unit_test)
  target_sources(yolo26-tiny-drone-tracker_unit_test PRIVATE ${DRONE_TRACKER_UTILS})
  target_link_libraries(yolo26-tiny-drone-tracker_unit_test PRIVATE SimaNeatApps::sima_neat)
endif()
