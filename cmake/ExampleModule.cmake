function(_sima_neat_apps_find_sources out_var module_dir)
  set(_sources)
  if (EXISTS "${module_dir}/main.cpp")
    list(APPEND _sources "${module_dir}/main.cpp")
  else()
    file(GLOB _sources CONFIGURE_DEPENDS
      "${module_dir}/*.cpp"
      "${module_dir}/*.cc"
      "${module_dir}/*.cxx")
  endif()

  if (NOT _sources)
    message(FATAL_ERROR "No C++ sources found under ${module_dir}")
  endif()

  set(${out_var} "${_sources}" PARENT_SCOPE)
endfunction()

function(_sima_neat_apps_ensure_neat_target apps_root)
  if (TARGET SimaNeat::sima_neat)
    return()
  endif()

  find_package(SimaNeat CONFIG QUIET)
  if (TARGET SimaNeat::sima_neat)
    return()
  endif()

  set(_core_root "${apps_root}/../core")
  set(_core_lib "${_core_root}/build/libsima_neat.a")
  set(_core_include "${_core_root}/include")
  if (NOT EXISTS "${_core_lib}" OR NOT EXISTS "${_core_include}")
    message(FATAL_ERROR
      "Could not find a usable SimaNeat package or local core build. "
      "Expected ${_core_lib} and ${_core_include}.")
  endif()

  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GST REQUIRED IMPORTED_TARGET
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0
    gstreamer-sdp-1.0
    gstreamer-rtsp-server-1.0
    glib-2.0
  )
  pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)

  add_library(SimaNeat::sima_neat STATIC IMPORTED GLOBAL)
  set_target_properties(SimaNeat::sima_neat PROPERTIES
    IMPORTED_LOCATION "${_core_lib}"
    INTERFACE_COMPILE_DEFINITIONS "SIMA_WITH_OPENCV;SIMA_HAS_SIMAAI_POOL=1"
    INTERFACE_INCLUDE_DIRECTORIES "${_core_include};${_core_include}/pipeline"
    INTERFACE_LINK_LIBRARIES "PkgConfig::GST;PkgConfig::OPENCV;gstsimaallocator;gstsimaaibufferpool"
  )
endfunction()

function(_sima_neat_apps_ensure_support_runtime apps_root)
  if (TARGET SimaNeatApps::support_runtime)
    return()
  endif()

  find_package(nlohmann_json REQUIRED)

  add_library(sima_neat_apps_support_runtime STATIC
    "${apps_root}/support/runtime/asset_utils.cpp"
    "${apps_root}/support/runtime/example_utils.cpp"
    "${apps_root}/support/object_detection/obj_detection_utils.cpp"
  )
  add_library(SimaNeatApps::support_runtime ALIAS sima_neat_apps_support_runtime)

  target_link_libraries(sima_neat_apps_support_runtime
    PUBLIC
      SimaNeat::sima_neat
      nlohmann_json::nlohmann_json
  )

  target_include_directories(sima_neat_apps_support_runtime
    PUBLIC
      "${apps_root}"
  )
endfunction()

function(_sima_neat_apps_ensure_support_optiview apps_root)
  if (TARGET SimaNeatApps::support_optiview)
    return()
  endif()

  _sima_neat_apps_ensure_support_runtime("${apps_root}")

  add_library(sima_neat_apps_support_optiview STATIC
    "${apps_root}/support/optiview/graphpipes_optiview_helpers.cpp"
  )
  add_library(SimaNeatApps::support_optiview ALIAS sima_neat_apps_support_optiview)

  target_link_libraries(sima_neat_apps_support_optiview
    PUBLIC
      SimaNeatApps::support_runtime
      SimaNeat::sima_neat
  )

  target_include_directories(sima_neat_apps_support_optiview
    PUBLIC
      "${apps_root}"
  )
endfunction()

function(_sima_neat_apps_ensure_support_testing apps_root)
  if (TARGET SimaNeatApps::support_testing)
    return()
  endif()

  find_package(nlohmann_json REQUIRED)

  add_library(sima_neat_apps_support_testing STATIC
    "${apps_root}/support/testing/optiview_json_listener.cpp"
    "${apps_root}/support/testing/test_process.cpp"
  )
  add_library(SimaNeatApps::support_testing ALIAS sima_neat_apps_support_testing)

  target_link_libraries(sima_neat_apps_support_testing
    PUBLIC
      nlohmann_json::nlohmann_json
  )

  target_include_directories(sima_neat_apps_support_testing
    PUBLIC
      "${apps_root}"
  )
endfunction()

function(_sima_neat_apps_add_optiview_e2e_test example_name module_dir example_target)
  get_filename_component(_apps_root "${module_dir}/../../../.." ABSOLUTE)
  set(_e2e_source "${module_dir}/tests/e2e_test.cpp")
  if (NOT EXISTS "${_e2e_source}")
    message(FATAL_ERROR
      "Example ${example_name} requested standard e2e testing but ${_e2e_source} does not exist.")
  endif()

  set(_e2e_target "${example_name}_e2e_test")
  add_executable("${_e2e_target}" "${_e2e_source}")
  target_link_libraries("${_e2e_target}"
    PRIVATE
      SimaNeatApps::support_testing
  )
  target_include_directories("${_e2e_target}"
    PRIVATE
      "${_apps_root}"
  )

  add_test(
    NAME "${example_name}.optiview_json_e2e"
    COMMAND $<TARGET_FILE:${_e2e_target}> $<TARGET_FILE:${example_target}>
  )
  set_tests_properties("${example_name}.optiview_json_e2e" PROPERTIES
    SKIP_RETURN_CODE 77
    LABELS "e2e"
    WORKING_DIRECTORY "${_apps_root}"
  )
endfunction()

# ---------------------------------------------------------------------------
# Generic unit test: compile tests/unit_test.cpp, register with label "unit".
# ---------------------------------------------------------------------------
function(_sima_neat_apps_add_unit_test example_name module_dir example_target)
  get_filename_component(_apps_root "${module_dir}/../../../.." ABSOLUTE)
  set(_unit_source "${module_dir}/tests/unit_test.cpp")
  if (NOT EXISTS "${_unit_source}")
    message(FATAL_ERROR
      "Example ${example_name} requested UNIT_TEST but ${_unit_source} does not exist.")
  endif()

  set(_unit_target "${example_name}_unit_test")
  add_executable("${_unit_target}" "${_unit_source}")
  target_link_libraries("${_unit_target}"
    PRIVATE
      SimaNeatApps::support_testing
  )
  target_include_directories("${_unit_target}"
    PRIVATE
      "${_apps_root}"
  )

  add_test(
    NAME "${example_name}.unit"
    COMMAND $<TARGET_FILE:${_unit_target}> $<TARGET_FILE:${example_target}>
  )
  set_tests_properties("${example_name}.unit" PROPERTIES
    SKIP_RETURN_CODE 77
    LABELS "unit"
    WORKING_DIRECTORY "${_apps_root}"
  )
endfunction()

# ---------------------------------------------------------------------------
# Generic e2e test: compile tests/e2e_test.cpp, register with label "e2e".
# ---------------------------------------------------------------------------
function(_sima_neat_apps_add_generic_e2e_test example_name module_dir example_target)
  get_filename_component(_apps_root "${module_dir}/../../../.." ABSOLUTE)
  set(_e2e_source "${module_dir}/tests/e2e_test.cpp")
  if (NOT EXISTS "${_e2e_source}")
    message(FATAL_ERROR
      "Example ${example_name} requested E2E_TEST but ${_e2e_source} does not exist.")
  endif()

  set(_e2e_target "${example_name}_e2e_test")
  add_executable("${_e2e_target}" "${_e2e_source}")
  target_link_libraries("${_e2e_target}"
    PRIVATE
      SimaNeatApps::support_testing
  )
  target_include_directories("${_e2e_target}"
    PRIVATE
      "${_apps_root}"
  )

  add_test(
    NAME "${example_name}.e2e"
    COMMAND $<TARGET_FILE:${_e2e_target}> $<TARGET_FILE:${example_target}>
  )
  set_tests_properties("${example_name}.e2e" PROPERTIES
    SKIP_RETURN_CODE 77
    LABELS "e2e"
    WORKING_DIRECTORY "${_apps_root}"
  )
endfunction()

function(sima_neat_apps_module example_name)
  set(options OPTIVIEW OPTIVIEW_E2E_TEST UNIT_TEST E2E_TEST)
  set(one_value_args OUTPUT_TARGET_VAR)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(APP "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if (COMMAND sima_neat_apps_add_example)
    if (APP_OPTIVIEW)
      sima_neat_apps_add_example("${example_name}" OPTIVIEW OUTPUT_TARGET_VAR _module_target_name)
    else()
      sima_neat_apps_add_example("${example_name}" OUTPUT_TARGET_VAR _module_target_name)
    endif()
    if (APP_OUTPUT_TARGET_VAR)
      set(${APP_OUTPUT_TARGET_VAR} "${_module_target_name}" PARENT_SCOPE)
    endif()

    get_filename_component(_module_dir "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
    if (BUILD_TESTING AND APP_OPTIVIEW_E2E_TEST)
      _sima_neat_apps_add_optiview_e2e_test("${example_name}" "${_module_dir}" "${_module_target_name}")
    endif()
    if (BUILD_TESTING AND APP_UNIT_TEST)
      _sima_neat_apps_add_unit_test("${example_name}" "${_module_dir}" "${_module_target_name}")
    endif()
    if (BUILD_TESTING AND APP_E2E_TEST)
      _sima_neat_apps_add_generic_e2e_test("${example_name}" "${_module_dir}" "${_module_target_name}")
    endif()

    return()
  endif()

  get_filename_component(_module_dir "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
  get_filename_component(_apps_root "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

  if (APP_SOURCES)
    set(_sources "${APP_SOURCES}")
  else()
    _sima_neat_apps_find_sources(_sources "${_module_dir}")
  endif()

  _sima_neat_apps_ensure_neat_target("${_apps_root}")
  _sima_neat_apps_ensure_support_runtime("${_apps_root}")
  if (APP_OPTIVIEW)
    _sima_neat_apps_ensure_support_optiview("${_apps_root}")
  endif()
  if (BUILD_TESTING)
    _sima_neat_apps_ensure_support_testing("${_apps_root}")
  endif()

  add_executable(${example_name} ${_sources})

  target_link_libraries(${example_name}
    PRIVATE
      SimaNeat::sima_neat
      SimaNeatApps::support_runtime
  )

  if (APP_OPTIVIEW)
    target_link_libraries(${example_name} PRIVATE SimaNeatApps::support_optiview)
  endif()

  target_include_directories(${example_name}
    PRIVATE
      "${_apps_root}"
  )

  if (APP_OUTPUT_TARGET_VAR)
    set(${APP_OUTPUT_TARGET_VAR} "${example_name}" PARENT_SCOPE)
  endif()

  if (BUILD_TESTING AND APP_OPTIVIEW_E2E_TEST)
    _sima_neat_apps_add_optiview_e2e_test("${example_name}" "${_module_dir}" "${example_name}")
  endif()

  if (BUILD_TESTING AND APP_UNIT_TEST)
    _sima_neat_apps_add_unit_test("${example_name}" "${_module_dir}" "${example_name}")
  endif()

  if (BUILD_TESTING AND APP_E2E_TEST)
    _sima_neat_apps_add_generic_e2e_test("${example_name}" "${_module_dir}" "${example_name}")
  endif()
endfunction()
