include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/YamlCpp.cmake")
sima_neat_apps_ensure_yaml_cpp()

add_library(sima_neat_apps_support_runtime STATIC
  sima_examples/anomaly_regions.cpp
  sima_examples/asset_utils.cpp
  sima_examples/config.cpp
  sima_examples/example_utils.cpp
  sima_examples/obj_detection_utils.cpp
  sima_examples/rtsp_probe.cpp
)
add_library(SimaNeatApps::support_runtime ALIAS sima_neat_apps_support_runtime)

target_link_libraries(sima_neat_apps_support_runtime
  PUBLIC
    SimaNeatApps::sima_neat
    opencv_videoio
)

target_link_libraries(sima_neat_apps_support_runtime
  PUBLIC
    nlohmann_json::nlohmann_json
    SimaNeatApps::yaml_cpp
)

target_include_directories(sima_neat_apps_support_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
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
    sima_examples/config.cpp
    sima_examples/testing/metadata_json_listener.cpp
    sima_examples/testing/source_cases.cpp
    sima_examples/testing/test_checks.cpp
    sima_examples/testing/test_config.cpp
    sima_examples/testing/test_process.cpp
  )
  add_library(SimaNeatApps::support_testing ALIAS sima_neat_apps_support_testing)
  target_link_libraries(sima_neat_apps_support_testing
    PUBLIC
      SimaNeatApps::support_runtime
      nlohmann_json::nlohmann_json
      PkgConfig::OPENCV
      SimaNeatApps::yaml_cpp
  )
  target_include_directories(sima_neat_apps_support_testing
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()
