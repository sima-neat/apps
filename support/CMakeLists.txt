add_library(sima_neat_apps_support_runtime STATIC
  runtime/asset_utils.cpp
  runtime/example_utils.cpp
  object_detection/obj_detection_utils.cpp
)
add_library(SimaNeatApps::support_runtime ALIAS sima_neat_apps_support_runtime)

target_link_libraries(sima_neat_apps_support_runtime
  PUBLIC
    SimaNeat::sima_neat
)

target_link_libraries(sima_neat_apps_support_runtime PUBLIC nlohmann_json::nlohmann_json)

target_include_directories(sima_neat_apps_support_runtime
  PUBLIC
    ${PROJECT_SOURCE_DIR}
)

if (SIMANEAT_APPS_BUILD_OPTIVIEW_SUPPORT)
  add_library(sima_neat_apps_support_optiview STATIC
    optiview/graphpipes_optiview_helpers.cpp
  )
  add_library(SimaNeatApps::support_optiview ALIAS sima_neat_apps_support_optiview)
  target_link_libraries(sima_neat_apps_support_optiview
    PUBLIC
      SimaNeatApps::support_runtime
      SimaNeat::sima_neat
  )
  target_include_directories(sima_neat_apps_support_optiview
    PUBLIC
      ${PROJECT_SOURCE_DIR}
  )
endif()

if (BUILD_TESTING)
  add_library(sima_neat_apps_support_testing STATIC
    testing/optiview_json_listener.cpp
  )
  add_library(SimaNeatApps::support_testing ALIAS sima_neat_apps_support_testing)
  target_link_libraries(sima_neat_apps_support_testing
    PUBLIC
      nlohmann_json::nlohmann_json
  )
  target_include_directories(sima_neat_apps_support_testing
    PUBLIC
      ${PROJECT_SOURCE_DIR}
  )
endif()
