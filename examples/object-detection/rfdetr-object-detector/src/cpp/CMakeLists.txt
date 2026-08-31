cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rfdetr-object-detector LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "rfdetr-object-detector"
  OUTPUT_TARGET_VAR RFDETR_TARGET
  METADATA_E2E_TEST
)

if (BUILD_TESTING)
  get_filename_component(APPS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
  add_executable(
    rfdetr-object-detector_unit_test
    "${CMAKE_CURRENT_LIST_DIR}/../../tests/cpp/test_unit.cpp"
  )
  target_link_libraries(
    rfdetr-object-detector_unit_test
    PRIVATE SimaNeatApps::support_runtime SimaNeatApps::support_testing
  )
  target_include_directories(rfdetr-object-detector_unit_test PRIVATE "${APPS_ROOT}")
  add_test(
    NAME rfdetr-object-detector.unit
    COMMAND $<TARGET_FILE:rfdetr-object-detector_unit_test> $<TARGET_FILE:${RFDETR_TARGET}>
  )
  set_tests_properties(
    rfdetr-object-detector.unit
    PROPERTIES SKIP_RETURN_CODE 77 LABELS "unit" WORKING_DIRECTORY "${APPS_ROOT}"
  )
endif()
