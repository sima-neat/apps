add_library(sima_neat_apps_support_runtime STATIC
  runtime/asset_utils.cpp
  runtime/config_utils.cpp
  runtime/example_utils.cpp
  object_detection/obj_detection_utils.cpp
)
add_library(SimaNeatApps::support_runtime ALIAS sima_neat_apps_support_runtime)

target_link_libraries(sima_neat_apps_support_runtime
  PUBLIC
    SimaNeatApps::sima_neat
)

target_link_libraries(sima_neat_apps_support_runtime PUBLIC nlohmann_json::nlohmann_json)

target_include_directories(sima_neat_apps_support_runtime
  PUBLIC
    ${PROJECT_SOURCE_DIR}
)

if (BUILD_TESTING)
  # The e2e assertions decode saved frames, so the test support library needs
  # OpenCV in its own right. The example binaries already require it, through
  # the Neat core target, and this resolves it the same way the core fallback
  # does when there is no installed SDK.
  if (NOT TARGET PkgConfig::OPENCV)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)
  endif()

  add_library(sima_neat_apps_support_testing STATIC
    runtime/config_utils.cpp
    testing/metadata_json_listener.cpp
    testing/test_config.cpp
    testing/test_process.cpp
  )
  add_library(SimaNeatApps::support_testing ALIAS sima_neat_apps_support_testing)
  target_link_libraries(sima_neat_apps_support_testing
    PUBLIC
      nlohmann_json::nlohmann_json
      PkgConfig::OPENCV
  )
  target_include_directories(sima_neat_apps_support_testing
    PUBLIC
      ${PROJECT_SOURCE_DIR}
  )
endif()
