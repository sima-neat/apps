foreach(example IN ITEMS
    detr-object-detector
    faster-rcnn-object-detector
    single-stream-object-detector
    yolo26-object-detector
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/src/cpp/CMakeLists.txt")
    add_subdirectory(${example}/src/cpp ${example})
  endif()
endforeach()
add_subdirectory(multi-stream-object-detector/src/cpp multi-stream-object-detector)
