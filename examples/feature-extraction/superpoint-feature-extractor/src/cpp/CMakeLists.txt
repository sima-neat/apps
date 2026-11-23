cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(superpoint-feature-extractor LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "superpoint-feature-extractor"
  OUTPUT_TARGET_VAR SUPERPOINT_TARGET
  UNIT_TEST
  E2E_TEST
)

set(SUPERPOINT_UTILS "${CMAKE_CURRENT_LIST_DIR}/utils/feature_points.cpp")
target_sources(${SUPERPOINT_TARGET} PRIVATE ${SUPERPOINT_UTILS})

# The unit test checks the feature-point mapping in process, so it compiles the utils too, and
# links the Neat Library their floats() reads tensors with.
if (TARGET superpoint-feature-extractor_unit_test)
  target_sources(superpoint-feature-extractor_unit_test PRIVATE ${SUPERPOINT_UTILS})
  target_link_libraries(superpoint-feature-extractor_unit_test PRIVATE SimaNeatApps::sima_neat)
endif()
