foreach(example IN ITEMS
    detr-object-detection
    single-rtsp-object-detection-insight
    simple-object-detection-overlay-pipeline
    yolo26-object-detection-overlay
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/cpp/CMakeLists.txt")
    add_subdirectory(${example}/cpp ${example})
  endif()
endforeach()
add_subdirectory(multistream-object-detection-insight/cpp multistream-object-detection-insight)
