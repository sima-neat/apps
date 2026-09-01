cmake_minimum_required(VERSION 3.16)

# ── Stand-alone build support ─────────────────────────────────────────────────
# When built as its own project (not part of the apps tree), bootstrap the Neat
# targets and support libraries using the ExampleModule.cmake helpers.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(face-recognizer LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

# Locate the apps root regardless of whether we are standalone or nested.
get_filename_component(_SELF_DIR  "${CMAKE_CURRENT_LIST_DIR}"  ABSOLUTE)
# src/cpp  →  face-recognizer  →  face-recognition  →  examples  →  apps
get_filename_component(_APPS_ROOT "${_SELF_DIR}/../../../../.." ABSOLUTE)
get_filename_component(_MOD_ROOT  "${_SELF_DIR}/../.."          ABSOLUTE)  # face-recognizer/

# ── Bootstrap Neat targets (only needed when standalone) ──────────────────────
if (NOT TARGET SimaNeatApps::sima_neat)
  include("${_APPS_ROOT}/cmake/ExampleModule.cmake")
  _sima_neat_apps_ensure_neat_target("${_APPS_ROOT}")
  _sima_neat_apps_ensure_support_runtime("${_APPS_ROOT}")
endif()

# ── Shared library of processing nodes ───────────────────────────────────────
# All three executables link this.  Avoids duplicate compilation of the heavy
# SCRFD decode / alignment code.
add_library(face_recog_nodes STATIC
  "${_SELF_DIR}/scrfd_decode.cpp"
  "${_SELF_DIR}/align.cpp"
  "${_SELF_DIR}/gallery.cpp"
  "${_SELF_DIR}/match.cpp"
  "${_SELF_DIR}/overlay.cpp"
)
target_link_libraries(face_recog_nodes
  PUBLIC
    SimaNeatApps::sima_neat
    SimaNeatApps::support_runtime
)
target_include_directories(face_recog_nodes
  PUBLIC
    "${_SELF_DIR}"
    "${_APPS_ROOT}"
)

# ── Helper macro: create one executable ──────────────────────────────────────
macro(_fr_executable _name _source)
  add_executable(${_name} "${_source}")
  target_link_libraries(${_name}
    PRIVATE
      face_recog_nodes
      SimaNeatApps::sima_neat
      SimaNeatApps::support_runtime
  )
  target_include_directories(${_name}
    PRIVATE
      "${_SELF_DIR}"
      "${_APPS_ROOT}"
  )
  target_compile_definitions(${_name}
    PRIVATE
      SIMANEAT_APPS_EXAMPLE_SOURCE_DIR="${_SELF_DIR}"
  )
endmacro()

# ── face-recognizer (includes enrollment mode via --enroll flag) ──────────────
# Enrollment logic lives in tools/enroll.cpp and is dispatched from main() when
# --enroll is the first argument, so the single installed binary covers both
# recognition and gallery enrollment without a separate tool.
add_executable(face-recognizer
    "${_SELF_DIR}/main.cpp"
    "${_MOD_ROOT}/tools/enroll.cpp"
)
target_link_libraries(face-recognizer
    PRIVATE
        face_recog_nodes
        SimaNeatApps::sima_neat
        SimaNeatApps::support_runtime
)
target_include_directories(face-recognizer
    PRIVATE "${_SELF_DIR}" "${_APPS_ROOT}"
)
target_compile_definitions(face-recognizer
    PRIVATE SIMANEAT_APPS_EXAMPLE_SOURCE_DIR="${_SELF_DIR}"
)

# ── Tests ─────────────────────────────────────────────────────────────────────
if (BUILD_TESTING)
  if (NOT TARGET SimaNeatApps::support_testing)
    include("${_APPS_ROOT}/cmake/ExampleModule.cmake")
    _sima_neat_apps_ensure_support_testing("${_APPS_ROOT}")
  endif()

  # Unit test: in-process component tests (gallery, match, align, scrfd).
  # test_unit.cpp supplies its own main() — no support_testing spawn needed.
  set(_UNIT_SRC "${_MOD_ROOT}/tests/cpp/test_unit.cpp")
  if (EXISTS "${_UNIT_SRC}")
    add_executable(face-recognizer_unit_test "${_UNIT_SRC}")
    target_include_directories(face-recognizer_unit_test
      PRIVATE "${_SELF_DIR}" "${_APPS_ROOT}"
    )
    target_link_libraries(face-recognizer_unit_test
      PRIVATE face_recog_nodes
    )
    add_test(NAME face-recognizer.unit
             COMMAND face-recognizer_unit_test
             WORKING_DIRECTORY "${_APPS_ROOT}")
    set_tests_properties(face-recognizer.unit PROPERTIES
      SKIP_RETURN_CODE 77
      LABELS           "unit")
  endif()

  # E2E test: launches the binary with a real SCRFD+ArcFace model and
  # optional gallery.bin / video input.  Skips (exit 77) when assets are absent.
  set(_E2E_SRC "${_MOD_ROOT}/tests/cpp/test_e2e.cpp")
  if (EXISTS "${_E2E_SRC}")
    add_executable(face-recognizer_e2e_test "${_E2E_SRC}")
    target_link_libraries(face-recognizer_e2e_test
      PRIVATE SimaNeatApps::support_testing
    )
    target_include_directories(face-recognizer_e2e_test
      PRIVATE "${_APPS_ROOT}"
    )
    add_test(NAME face-recognizer.e2e
             COMMAND face-recognizer_e2e_test $<TARGET_FILE:face-recognizer>
             WORKING_DIRECTORY "${_APPS_ROOT}")
    set_tests_properties(face-recognizer.e2e PROPERTIES
      SKIP_RETURN_CODE 77
      LABELS           "e2e")
  endif()
endif()
