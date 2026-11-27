cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(multi-stream-multi-model LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "multi-stream-multi-model"
  OUTPUT_TARGET_VAR MULTI_STREAM_MULTI_MODEL_TARGET
  UNIT_TEST
  E2E_TEST
)

set(MULTI_STREAM_MULTI_MODEL_UTILS
  "${CMAKE_CURRENT_LIST_DIR}/utils/insight_metadata.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/utils/pending_results.cpp"
)
target_sources(${MULTI_STREAM_MULTI_MODEL_TARGET} PRIVATE ${MULTI_STREAM_MULTI_MODEL_UTILS})

# The unit test checks the metadata and the pending-result deadline in process, so it compiles the
# utils too, and links the Neat Library whose Box and OpenCV the metadata reads.
if (TARGET multi-stream-multi-model_unit_test)
  target_sources(multi-stream-multi-model_unit_test PRIVATE ${MULTI_STREAM_MULTI_MODEL_UTILS})
  target_link_libraries(multi-stream-multi-model_unit_test PRIVATE SimaNeatApps::sima_neat)
endif()
