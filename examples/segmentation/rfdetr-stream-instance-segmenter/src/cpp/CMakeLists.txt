cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rfdetr-stream-instance-segmenter LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")
sima_neat_apps_module("rfdetr-stream-instance-segmenter" UNIT_TEST E2E_TEST
  OUTPUT_TARGET_VAR _rfdetr_stream_instance_segmenter_target)

# The compiled model ships its top-k gather stage as a standalone TVM module (not a SimaNeat
# graph node), so this example additionally drives it in-process via the TVM C++ runtime.
set(SYSROOT "$ENV{SYSROOT}")
find_library(TVM_RUNTIME_LIB
  NAMES tvm_runtime
  HINTS "${SYSROOT}/usr/lib" "${SYSROOT}/usr/lib/aarch64-linux-gnu")
find_path(TVM_INCLUDE_DIR
  NAMES tvm/runtime/module.h
  HINTS "${SYSROOT}/usr/include")
find_path(DLPACK_INCLUDE_DIR
  NAMES dlpack/dlpack.h
  HINTS "${SYSROOT}/usr/include")

if (NOT TVM_RUNTIME_LIB OR NOT TVM_INCLUDE_DIR OR NOT DLPACK_INCLUDE_DIR)
  message(FATAL_ERROR
    "rfdetr-stream-instance-segmenter requires the TVM C++ runtime "
    "(tvm_runtime, tvm/runtime/module.h, dlpack/dlpack.h) from the Neat SDK sysroot.")
endif()

target_include_directories(${_rfdetr_stream_instance_segmenter_target} PRIVATE
  "${TVM_INCLUDE_DIR}" "${DLPACK_INCLUDE_DIR}")
target_link_libraries(${_rfdetr_stream_instance_segmenter_target} PRIVATE "${TVM_RUNTIME_LIB}")
