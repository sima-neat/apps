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
  UNIT_TEST
  E2E_TEST
)

# The unit test includes utils/feature_points.h, whose floats() reads tensors with the Neat
# Library.
if (TARGET superpoint-feature-extractor_unit_test)
  target_link_libraries(superpoint-feature-extractor_unit_test PRIVATE SimaNeatApps::sima_neat)
endif()
