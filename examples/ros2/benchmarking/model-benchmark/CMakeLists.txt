cmake_minimum_required(VERSION 3.16)
project(neat_model_benchmark)
find_package(ament_cmake REQUIRED)
install(PROGRAMS src/python/benchmark_node.py
        DESTINATION lib/${PROJECT_NAME})
install(FILES src/python/model_support.py
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch src/common DESTINATION share/${PROJECT_NAME}
        PATTERN "__pycache__" EXCLUDE)
ament_package()
