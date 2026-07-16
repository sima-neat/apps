cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(fastsam-multistream LANGUAGES CXX)
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

# FastSAM + MobileCLIP drive GStreamer/OpenCV and nlohmann_json directly, so the module
# links them explicitly rather than relying on the SimaNeat package to re-export them.
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET
  gstreamer-1.0
  gstreamer-app-1.0
  gstreamer-video-1.0
  gstreamer-sdp-1.0
  gstreamer-rtsp-server-1.0
  glib-2.0)
pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)
find_package(nlohmann_json REQUIRED)
find_package(Threads REQUIRED)

# The pipeline, FastSAM runner, and CLIP tower live beside main.cpp; compile them into a
# static library so both the example binary and any future tests can link them.
add_library(fastsam_multistream_lib STATIC
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fastsam.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/pipeline.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/clip/image_encoder.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/utils/tensors.cpp"
)

target_link_libraries(fastsam_multistream_lib
  PUBLIC
    SimaNeatApps::sima_neat
    SimaNeatApps::support_runtime
    nlohmann_json::nlohmann_json
    PkgConfig::GST
    PkgConfig::OPENCV
    Threads::Threads
)

target_include_directories(fastsam_multistream_lib
  PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}"
    "${APPS_ROOT}"
)

target_compile_definitions(fastsam_multistream_lib
  PUBLIC
    SIMA_WITH_OPENCV
    SIMA_HAS_SIMAAI_POOL=1
    FASTSAM_CPP_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}"
)

# Register the example binary through the shared framework helper (target name, output
# name, install layout). main.cpp is the only translation unit built here; the rest comes
# from fastsam_multistream_lib.
sima_neat_apps_module(
  "fastsam-multistream"
  OUTPUT_TARGET_VAR FASTSAM_MULTISTREAM_TARGET
  SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
)

target_link_libraries("${FASTSAM_MULTISTREAM_TARGET}"
  PRIVATE
    fastsam_multistream_lib
)
