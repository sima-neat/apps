cmake_minimum_required(VERSION 3.16)
project(neat_perception_switchboard)
find_package(ament_cmake REQUIRED)
install(PROGRAMS src/python/input_node.py src/python/yolo_node.py
                 src/python/depth_node.py src/python/vlm_node.py src/python/output_node.py
        DESTINATION lib/${PROJECT_NAME})
install(FILES src/python/image_transport.py src/python/visual_output.py
              src/python/yolo_masks.py src/python/overlays.py src/python/node_tools.py
              src/python/camera_image.py
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch src/common DESTINATION share/${PROJECT_NAME}
        PATTERN "__pycache__" EXCLUDE)
ament_package()
