add_executable(app_model_resnet50
  main.cpp
)

target_link_libraries(app_model_resnet50
  PRIVATE
    SimaNeat::sima_neat
    SimaNeatApps::support_runtime
)

target_include_directories(app_model_resnet50
  PRIVATE
    ${PROJECT_SOURCE_DIR}
)

set_target_properties(app_model_resnet50 PROPERTIES
  OUTPUT_NAME model_resnet50
)
