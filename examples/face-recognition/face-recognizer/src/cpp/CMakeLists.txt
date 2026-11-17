cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(face-recognizer LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "face-recognizer"
  UNIT_TEST
  E2E_TEST
  OUTPUT_TARGET_VAR FACE_RECOGNIZER_TARGET
)

set(FACE_RECOGNIZER_HELPERS
  "${CMAKE_CURRENT_LIST_DIR}/align.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/gallery.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/match.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/scrfd_decode.cpp"
)
target_sources(${FACE_RECOGNIZER_TARGET} PRIVATE ${FACE_RECOGNIZER_HELPERS})

# The unit test compiles the application's helpers, so it needs the application's build setup.
if (TARGET face-recognizer_unit_test)
  target_sources(face-recognizer_unit_test PRIVATE ${FACE_RECOGNIZER_HELPERS})
  target_compile_definitions(face-recognizer_unit_test PRIVATE
      $<TARGET_PROPERTY:${FACE_RECOGNIZER_TARGET},COMPILE_DEFINITIONS>)
  target_link_libraries(face-recognizer_unit_test PRIVATE SimaNeatApps::support_runtime)
endif()
