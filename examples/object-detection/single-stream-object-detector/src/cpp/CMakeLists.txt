cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(single-stream-object-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "single-stream-object-detector"
  OUTPUT_TARGET_VAR YOLO26_ONE_RTSP_INSIGHT_TARGET
  UNIT_TEST
  METADATA_E2E_TEST
)

# The helpers in utils/ are compiled with main.cpp. The vendored ones are copies of
# support/templates/cpp/, kept identical by scripts/sync_utils.py.
target_sources(${YOLO26_ONE_RTSP_INSIGHT_TARGET}
  PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/utils/app_config.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/utils/config.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/utils/detections.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/utils/detector.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/utils/frames.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/utils/insight.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/utils/sources.cpp"
)
# The helpers need nlohmann_json for Insight metadata and OpenCV's videoio for cv::VideoCapture,
# which probes RTSP streams. The Neat target does not bring videoio.
target_link_libraries(${YOLO26_ONE_RTSP_INSIGHT_TARGET}
  PRIVATE
    nlohmann_json::nlohmann_json
    opencv_videoio
)

if (BUILD_TESTING)
  add_test(
    NAME single-stream-object-detector.requires_rtsp
    COMMAND $<TARGET_FILE:${YOLO26_ONE_RTSP_INSIGHT_TARGET}> --config /nonexistent/single-rtsp-config.yaml
  )
  set_tests_properties(single-stream-object-detector.requires_rtsp PROPERTIES
    WILL_FAIL TRUE
    LABELS "unit"
  )
endif()

# The unit test exercises the vendored ffprobe command, NV16 conversion and pull-outcome
# helpers, which need the Neat headers.
if (TARGET single-stream-object-detector_unit_test)
  target_sources(single-stream-object-detector_unit_test
    PRIVATE
      "${CMAKE_CURRENT_LIST_DIR}/utils/frames.cpp"
      "${CMAKE_CURRENT_LIST_DIR}/utils/sources.cpp"
  )
  target_link_libraries(single-stream-object-detector_unit_test
    PRIVATE
      SimaNeatApps::sima_neat
      opencv_videoio
  )
endif()
