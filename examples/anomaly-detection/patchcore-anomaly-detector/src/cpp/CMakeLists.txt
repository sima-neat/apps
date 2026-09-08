cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(patchcore-anomaly-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")
sima_neat_apps_module("patchcore-anomaly-detector" UNIT_TEST E2E_TEST OUTPUT_TARGET_VAR patchcore_target)

# The memory-bank/coreset scoring implementation and its OpenSSL SHA-256 use
# are specific to this application, so they're compiled and linked here
# rather than added to the shared support runtime that every example links.
find_package(OpenSSL REQUIRED)
target_sources(${patchcore_target} PRIVATE patchcore_memory_bank.cpp)
target_link_libraries(${patchcore_target} PRIVATE OpenSSL::Crypto)
