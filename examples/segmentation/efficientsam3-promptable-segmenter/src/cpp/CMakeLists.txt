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
  METADATA_E2E_TEST
)
# The tokenizer reads the gzip-compressed CLIP vocabulary.
target_link_libraries(${EFFICIENTSAM3_TARGET} PRIVATE z)

if (BUILD_TESTING)
  get_filename_component(APPS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
  file(RELATIVE_PATH EFFICIENTSAM3_SOURCE_DIR "${APPS_ROOT}" "${CMAKE_CURRENT_LIST_DIR}")
  # The unit test compiles main.cpp in, to test its functions directly.
  add_executable(
    efficientsam3-promptable-segmenter_unit_test
    "${CMAKE_CURRENT_LIST_DIR}/../../tests/cpp/test_unit.cpp"
  )
  target_link_libraries(
    efficientsam3-promptable-segmenter_unit_test
    PRIVATE SimaNeatApps::support_runtime SimaNeatApps::support_testing z
  )
  target_include_directories(efficientsam3-promptable-segmenter_unit_test PRIVATE "${APPS_ROOT}")
  target_compile_definitions(
    efficientsam3-promptable-segmenter_unit_test
    PRIVATE SIMANEAT_APPS_EXAMPLE_SOURCE_DIR="${EFFICIENTSAM3_SOURCE_DIR}"
  )
  add_test(
    NAME efficientsam3-promptable-segmenter.unit
    COMMAND $<TARGET_FILE:efficientsam3-promptable-segmenter_unit_test>
            $<TARGET_FILE:${EFFICIENTSAM3_TARGET}>
  )
  set_tests_properties(
    efficientsam3-promptable-segmenter.unit
    PROPERTIES SKIP_RETURN_CODE 77 LABELS "unit" WORKING_DIRECTORY "${APPS_ROOT}"
  )
endif()
