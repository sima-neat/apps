cmake_minimum_required(VERSION 3.16)
project(neat_genai_studio)
find_package(ament_cmake REQUIRED)
install(PROGRAMS src/python/studio_node.py
        DESTINATION lib/${PROJECT_NAME})
install(FILES src/python/image_transport.py
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch src/common DESTINATION share/${PROJECT_NAME}
        PATTERN "__pycache__" EXCLUDE)
# Keep the backend's source-relative assets together; run.sh owns its interpreters.
install(PROGRAMS run.sh setup.sh DESTINATION share/${PROJECT_NAME})
install(DIRECTORY src/ DESTINATION share/${PROJECT_NAME}/src
        PATTERN "__pycache__" EXCLUDE)
ament_package()
