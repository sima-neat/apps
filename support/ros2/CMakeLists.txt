cmake_minimum_required(VERSION 3.16)
project(neat_apps)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
ament_python_install_package(neat_apps)
install(PROGRAMS scripts/app_node scripts/image_source scripts/live_source scripts/result_sink scripts/insight_sink
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME} PATTERN "__pycache__" EXCLUDE)
ament_package()
