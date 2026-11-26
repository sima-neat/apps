cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(patchcore-anomaly-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "patchcore-anomaly-detector"
  OUTPUT_TARGET_VAR PATCHCORE_TARGET
  UNIT_TEST
  E2E_TEST
)

# The memory bank hashes its files with OpenSSL's SHA-256.
find_package(OpenSSL REQUIRED)
set(PATCHCORE_UTILS "${CMAKE_CURRENT_LIST_DIR}/utils/memory_bank.cpp")
target_sources(${PATCHCORE_TARGET} PRIVATE ${PATCHCORE_UTILS})
target_link_libraries(${PATCHCORE_TARGET} PRIVATE OpenSSL::Crypto)

# The unit test checks the memory bank in process, so it compiles the utils too.
if (TARGET patchcore-anomaly-detector_unit_test)
  target_sources(patchcore-anomaly-detector_unit_test PRIVATE ${PATCHCORE_UTILS})
  target_link_libraries(patchcore-anomaly-detector_unit_test PRIVATE OpenSSL::Crypto)
endif()
