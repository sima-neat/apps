cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(yolo26-tiny-drone-tracker LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

get_filename_component(APPS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
if (NOT TARGET SimaNeatApps::sima_neat)
  _sima_neat_apps_ensure_neat_target("${APPS_ROOT}")
endif()
if (NOT TARGET SimaNeatApps::support_runtime)
  _sima_neat_apps_ensure_support_runtime("${APPS_ROOT}")
endif()

add_library(yolo26_tiny_drone_tracker_lib STATIC
  "${CMAKE_CURRENT_LIST_DIR}/utils/tracker.cpp"
)
target_link_libraries(yolo26_tiny_drone_tracker_lib
  PUBLIC
    SimaNeatApps::sima_neat
    SimaNeatApps::support_runtime
)
target_include_directories(yolo26_tiny_drone_tracker_lib
  PUBLIC
    "${APPS_ROOT}"
)

sima_neat_apps_module(
  "yolo26-tiny-drone-tracker"
  UNIT_TEST
  E2E_TEST
  OUTPUT_TARGET_VAR YOLO26_TINY_DRONE_TRACKER_TARGET
  SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
)

target_link_libraries("${YOLO26_TINY_DRONE_TRACKER_TARGET}"
  PRIVATE
    yolo26_tiny_drone_tracker_lib
)

if (TARGET yolo26-tiny-drone-tracker_unit_test)
  target_link_libraries(yolo26-tiny-drone-tracker_unit_test
    PRIVATE
      yolo26_tiny_drone_tracker_lib
  )
endif()
