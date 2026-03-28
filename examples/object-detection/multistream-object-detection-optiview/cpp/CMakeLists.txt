cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(multistream-object-detection-optiview LANGUAGES CXX)
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

add_library(multistream_object_detection_optiview_lib STATIC
  "${CMAKE_CURRENT_LIST_DIR}/utils/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/utils/image_utils.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/utils/model_family.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/utils/pipeline.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/utils/sample_utils.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/utils/workers.cpp"
)
target_link_libraries(multistream_object_detection_optiview_lib
  PUBLIC
    SimaNeat::sima_neat
    SimaNeatApps::support_runtime
)
target_include_directories(multistream_object_detection_optiview_lib
  PUBLIC
    "${APPS_ROOT}"
)
target_compile_definitions(multistream_object_detection_optiview_lib
  PUBLIC
    MULTISTREAM_OBJECT_DETECTION_OPTIVIEW_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}"
)

sima_neat_apps_module(
  "multistream-object-detection-optiview"
  UNIT_TEST
  E2E_TEST
  OUTPUT_TARGET_VAR MULTISTREAM_OBJECT_DETECTION_OPTIVIEW_TARGET
)

target_link_libraries("${MULTISTREAM_OBJECT_DETECTION_OPTIVIEW_TARGET}"
  PRIVATE
    multistream_object_detection_optiview_lib
)

if (TARGET multistream-object-detection-optiview_unit_test)
  target_link_libraries(multistream-object-detection-optiview_unit_test
    PRIVATE
      multistream_object_detection_optiview_lib
  )
endif()

foreach(target_name
    multistream_object_detection_optiview_lib
    "${MULTISTREAM_OBJECT_DETECTION_OPTIVIEW_TARGET}"
    multistream-object-detection-optiview_unit_test
    multistream-object-detection-optiview_e2e_test)
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
  if (TEST multistream-object-detection-optiview.unit)
    set_tests_properties(multistream-object-detection-optiview.unit PROPERTIES
      ENVIRONMENT "${MODALIX_TEST_ENV}"
    )
  endif()
  if (TEST multistream-object-detection-optiview.e2e)
    set_tests_properties(multistream-object-detection-optiview.e2e PROPERTIES
      ENVIRONMENT "${MODALIX_TEST_ENV}"
    )
  endif()
endif()
