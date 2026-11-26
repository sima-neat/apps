cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(efficientsam3-promptable-segmenter LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "efficientsam3-promptable-segmenter"
  OUTPUT_TARGET_VAR EFFICIENTSAM3_TARGET
  UNIT_TEST
  METADATA_E2E_TEST
)

set(EFFICIENTSAM3_UTILS "${CMAKE_CURRENT_LIST_DIR}/utils/efficientsam3.cpp")
target_sources(${EFFICIENTSAM3_TARGET} PRIVATE ${EFFICIENTSAM3_UTILS})
# The tokenizer reads the gzip-compressed CLIP vocabulary.
target_link_libraries(${EFFICIENTSAM3_TARGET} PRIVATE z)

# The unit test checks the tokenizer and the host work in process, so it compiles the utils too.
if (TARGET efficientsam3-promptable-segmenter_unit_test)
  target_sources(efficientsam3-promptable-segmenter_unit_test PRIVATE ${EFFICIENTSAM3_UTILS})
  target_link_libraries(efficientsam3-promptable-segmenter_unit_test PRIVATE z)
endif()
