cmake_minimum_required(VERSION 3.16)
project(neat_detection_vlm)
find_package(ament_cmake REQUIRED)
install(PROGRAMS src/python/input_node.py src/python/detection_node.py
                 src/python/vlm_node.py src/python/genai_server.py
        DESTINATION lib/${PROJECT_NAME})
install(FILES src/python/app_config.py src/python/box_decode.py
              src/python/image_transport.py src/python/visual_output.py
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch src/common DESTINATION share/${PROJECT_NAME})
ament_package()
