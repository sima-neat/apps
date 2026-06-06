cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(single-stream-object-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/ExampleModule.cmake")

set(MODALIX_TOOLCHAIN_LIB_DIR "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu")
set(MODALIX_TOOLCHAIN_USR_LIB_DIR "/opt/toolchain/aarch64/modalix/usr/lib")
set(MODALIX_NEAT_PLUGIN_DIR "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu/neat/gst-plugins")
set(MODALIX_NEAT_RUNTIME_LIB_DIR "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu/neat/runtime")
set(MODALIX_SIMA_NEAT_PLUGIN_LIB_DIR "/opt/toolchain/aarch64/modalix/usr/lib/sima-neat/gst-plugins")

sima_neat_apps_module(
  "single-stream-object-detector"
  OUTPUT_TARGET_VAR YOLOV8_ONE_RTSP_INSIGHT_TARGET
  UNIT_TEST
  METADATA_E2E_TEST
)

if (BUILD_TESTING)
  add_test(
    NAME single-stream-object-detector.requires_rtsp
    COMMAND $<TARGET_FILE:${YOLOV8_ONE_RTSP_INSIGHT_TARGET}> --config /nonexistent/single-rtsp-config.yaml
  )
  set_tests_properties(single-stream-object-detector.requires_rtsp PROPERTIES
    WILL_FAIL TRUE
    LABELS "unit"
  )
endif()

foreach(target_name
    "${YOLOV8_ONE_RTSP_INSIGHT_TARGET}"
    single-stream-object-detector_unit_test
    single-stream-object-detector_e2e_test)
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
  foreach(test_name
      single-stream-object-detector.metadata_json_e2e
      single-stream-object-detector.unit
      single-stream-object-detector.requires_rtsp)
    if (TEST "${test_name}")
      set_tests_properties("${test_name}" PROPERTIES
        ENVIRONMENT "${MODALIX_TEST_ENV}"
      )
    endif()
  endforeach()
endif()
