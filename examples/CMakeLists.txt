function(sima_neat_apps_add_example example_name)
  set(options OPTIVIEW)
  set(one_value_args OUTPUT_TARGET_VAR)
  cmake_parse_arguments(APP "${options}" "${one_value_args}" "" ${ARGN})

  set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
  if (NOT EXISTS "${source_path}")
    set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/${example_name}/main.cpp")
  endif()
  if (NOT EXISTS "${source_path}")
    message(FATAL_ERROR "Missing example source: ${source_path}")
  endif()

  string(MAKE_C_IDENTIFIER "${CMAKE_CURRENT_SOURCE_DIR}_${example_name}" target_suffix)
  set(target_name "app_${target_suffix}")

  add_executable("${target_name}" "${source_path}")

  target_link_libraries("${target_name}"
    PRIVATE
      SimaNeat::sima_neat
      SimaNeatApps::support_runtime
  )

  if (APP_OPTIVIEW)
    if (NOT TARGET SimaNeatApps::support_optiview)
      message(FATAL_ERROR
        "Example ${example_name} requires SimaNeatApps::support_optiview but "
        "SIMANEAT_APPS_BUILD_OPTIVIEW_SUPPORT is OFF")
    endif()
    target_link_libraries("${target_name}" PRIVATE SimaNeatApps::support_optiview)
  endif()

  target_include_directories("${target_name}"
    PRIVATE
      ${PROJECT_SOURCE_DIR}
  )

  set_target_properties("${target_name}" PROPERTIES OUTPUT_NAME "${example_name}")

  if (APP_OUTPUT_TARGET_VAR)
    set(${APP_OUTPUT_TARGET_VAR} "${target_name}" PARENT_SCOPE)
  endif()
endfunction()

add_subdirectory(classification)
add_subdirectory(object-detection)
add_subdirectory(tracking)
add_subdirectory(semantic-segmentation)
add_subdirectory(instance-segmentation)
add_subdirectory(depth-estimation)
