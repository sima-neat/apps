foreach(example IN ITEMS
    yolov8_multi_rtsp_demo
    yolov8_simple_detection_pipeline
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/CMakeLists.txt")
    add_subdirectory(${example})
  endif()
endforeach()
