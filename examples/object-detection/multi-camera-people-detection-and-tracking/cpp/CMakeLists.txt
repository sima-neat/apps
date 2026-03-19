cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(multi-camera-people-detection-and-tracking LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/ExampleModule.cmake")

get_filename_component(APPS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(MODALIX_TOOLCHAIN_LIB_DIR "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu")
set(MODALIX_TOOLCHAIN_USR_LIB_DIR "/opt/toolchain/aarch64/modalix/usr/lib")
set(MODALIX_NEAT_PLUGIN_DIR "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu/neat/gst-plugins")
set(MODALIX_NEAT_RUNTIME_LIB_DIR "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu/neat/runtime")
set(MODALIX_SIMA_NEAT_PLUGIN_LIB_DIR "/opt/toolchain/aarch64/modalix/usr/lib/sima-neat/gst-plugins")

if (NOT TARGET SimaNeat::sima_neat)
  _sima_neat_apps_ensure_neat_target("${APPS_ROOT}")
endif()
if (NOT TARGET SimaNeatApps::support_runtime)
  _sima_neat_apps_ensure_support_runtime("${APPS_ROOT}")
endif()

add_library(multi_camera_people_tracking_lib STATIC
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/tracker.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sample_utils.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/pipeline.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/image_utils.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/workers.cpp"
)
target_link_libraries(multi_camera_people_tracking_lib
  PUBLIC
    SimaNeat::sima_neat
    SimaNeatApps::support_runtime
)
target_include_directories(multi_camera_people_tracking_lib
  PUBLIC
    "${APPS_ROOT}"
)
target_compile_definitions(multi_camera_people_tracking_lib
  PUBLIC
    MULTI_CAMERA_PEOPLE_TRACKING_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}"
)

sima_neat_apps_module(
  "multi-camera-people-detection-and-tracking"
  UNIT_TEST
  E2E_TEST
  OUTPUT_TARGET_VAR MULTI_CAMERA_PEOPLE_TRACKING_TARGET
  SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
)

target_link_libraries("${MULTI_CAMERA_PEOPLE_TRACKING_TARGET}"
  PRIVATE
    multi_camera_people_tracking_lib
)

if (TARGET multi-camera-people-detection-and-tracking_unit_test)
  target_link_libraries(multi-camera-people-detection-and-tracking_unit_test
    PRIVATE
      multi_camera_people_tracking_lib
  )
endif()

foreach(target_name
    multi_camera_people_tracking_lib
    "${MULTI_CAMERA_PEOPLE_TRACKING_TARGET}"
    multi-camera-people-detection-and-tracking_unit_test
    multi-camera-people-detection-and-tracking_e2e_test)
  if (TARGET "${target_name}" AND EXISTS "${MODALIX_TOOLCHAIN_LIB_DIR}")
    set_target_properties("${target_name}" PROPERTIES
      BUILD_RPATH "${MODALIX_TOOLCHAIN_LIB_DIR}"
      INSTALL_RPATH "${MODALIX_TOOLCHAIN_LIB_DIR}"
    )
  endif()
endforeach()

if (BUILD_TESTING AND EXISTS "${MODALIX_TOOLCHAIN_LIB_DIR}")
  set(MODALIX_TEST_LD_LIBRARY_PATH "${MODALIX_TOOLCHAIN_LIB_DIR}")
  if (EXISTS "${MODALIX_TOOLCHAIN_USR_LIB_DIR}")
    string(APPEND MODALIX_TEST_LD_LIBRARY_PATH ":${MODALIX_TOOLCHAIN_USR_LIB_DIR}")
  endif()
  if (EXISTS "${MODALIX_NEAT_RUNTIME_LIB_DIR}")
    string(APPEND MODALIX_TEST_LD_LIBRARY_PATH ":${MODALIX_NEAT_RUNTIME_LIB_DIR}")
  endif()
  if (EXISTS "${MODALIX_SIMA_NEAT_PLUGIN_LIB_DIR}")
    string(APPEND MODALIX_TEST_LD_LIBRARY_PATH ":${MODALIX_SIMA_NEAT_PLUGIN_LIB_DIR}")
  endif()
  string(APPEND MODALIX_TEST_LD_LIBRARY_PATH ":$ENV{LD_LIBRARY_PATH}")

  set(MODALIX_TEST_ENV "LD_LIBRARY_PATH=${MODALIX_TEST_LD_LIBRARY_PATH}")
  if (EXISTS "${MODALIX_NEAT_PLUGIN_DIR}")
    list(APPEND MODALIX_TEST_ENV "SIMA_GST_PLUGIN_DIR=${MODALIX_NEAT_PLUGIN_DIR}")
  endif()
  if (TEST multi-camera-people-detection-and-tracking.unit)
    set_tests_properties(multi-camera-people-detection-and-tracking.unit PROPERTIES
      ENVIRONMENT "${MODALIX_TEST_ENV}"
    )
  endif()
  if (TEST multi-camera-people-detection-and-tracking.e2e)
    set_tests_properties(multi-camera-people-detection-and-tracking.e2e PROPERTIES
      ENVIRONMENT "${MODALIX_TEST_ENV}"
    )
  endif()
endif()
