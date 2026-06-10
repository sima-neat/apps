cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(multi-stream-people-tracker LANGUAGES CXX)
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

add_library(multi_stream_people_tracker_lib STATIC
  "${CMAKE_CURRENT_LIST_DIR}/utils/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/utils/tracker.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/utils/sample_utils.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/utils/pipeline.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/utils/image_utils.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/utils/workers.cpp"
)
target_link_libraries(multi_stream_people_tracker_lib
  PUBLIC
    SimaNeatApps::sima_neat
    SimaNeatApps::support_runtime
)
target_include_directories(multi_stream_people_tracker_lib
  PUBLIC
    "${APPS_ROOT}"
)
target_compile_definitions(multi_stream_people_tracker_lib
  PUBLIC
    MULTI_CAMERA_PEOPLE_TRACKING_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}"
)

sima_neat_apps_module(
  "multi-stream-people-tracker"
  UNIT_TEST
  E2E_TEST
  OUTPUT_TARGET_VAR MULTI_CAMERA_PEOPLE_TRACKING_TARGET
  SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
)

target_link_libraries("${MULTI_CAMERA_PEOPLE_TRACKING_TARGET}"
  PRIVATE
    multi_stream_people_tracker_lib
)

if (TARGET multi-stream-people-tracker_unit_test)
  target_link_libraries(multi-stream-people-tracker_unit_test
    PRIVATE
      multi_stream_people_tracker_lib
  )
endif()
