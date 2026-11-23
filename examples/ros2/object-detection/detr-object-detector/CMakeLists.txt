cmake_minimum_required(VERSION 3.16)
project(neat_detr_object_detector)
find_package(ament_cmake REQUIRED)
install(PROGRAMS src/python/detection_node.py src/python/result_sink.py src/python/input_node.py
        DESTINATION lib/${PROJECT_NAME})
install(FILES src/python/model_support.py
              src/python/visual_output.py
              src/python/app_config.py
              src/python/image_transport.py
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch src/common DESTINATION share/${PROJECT_NAME}
        PATTERN "__pycache__" EXCLUDE)
ament_package()
