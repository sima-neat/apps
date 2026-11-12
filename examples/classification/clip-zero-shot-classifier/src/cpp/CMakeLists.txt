cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(clip-zero-shot-classifier LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "clip-zero-shot-classifier"
  OUTPUT_TARGET_VAR CLIP_ZERO_SHOT_TARGET
  METADATA_E2E_TEST
)

if (BUILD_TESTING)
  get_filename_component(APPS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
  file(RELATIVE_PATH CLIP_ZERO_SHOT_SOURCE_DIR "${APPS_ROOT}" "${CMAKE_CURRENT_LIST_DIR}")
  # The unit test compiles main.cpp in, to test its functions directly.
  add_executable(
    clip-zero-shot-classifier_unit_test
    "${CMAKE_CURRENT_LIST_DIR}/../../tests/cpp/test_unit.cpp"
  )
  target_link_libraries(
    clip-zero-shot-classifier_unit_test
    PRIVATE SimaNeatApps::support_runtime SimaNeatApps::support_testing
  )
  target_include_directories(clip-zero-shot-classifier_unit_test PRIVATE "${APPS_ROOT}")
  target_compile_definitions(
    clip-zero-shot-classifier_unit_test
    PRIVATE SIMANEAT_APPS_EXAMPLE_SOURCE_DIR="${CLIP_ZERO_SHOT_SOURCE_DIR}"
  )
  add_test(
    NAME clip-zero-shot-classifier.unit
    COMMAND $<TARGET_FILE:clip-zero-shot-classifier_unit_test>
            $<TARGET_FILE:${CLIP_ZERO_SHOT_TARGET}>
  )
  # The e2e test compiles main.cpp in too, for the panel layout it looks for in the received video.
  target_link_libraries(clip-zero-shot-classifier_e2e_test PRIVATE SimaNeatApps::support_runtime)
  target_compile_definitions(
    clip-zero-shot-classifier_e2e_test
    PRIVATE SIMANEAT_APPS_EXAMPLE_SOURCE_DIR="${CLIP_ZERO_SHOT_SOURCE_DIR}"
  )
  set_tests_properties(
    clip-zero-shot-classifier.unit
    PROPERTIES SKIP_RETURN_CODE 77 LABELS "unit" WORKING_DIRECTORY "${APPS_ROOT}"
  )
endif()
