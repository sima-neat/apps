foreach(example IN ITEMS
    single-image-object-detector
    single-stream-object-detector
    image-folder-object-detector
    yolo26-object-detector
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/cpp/CMakeLists.txt")
    add_subdirectory(${example}/cpp ${example})
  endif()
endforeach()
add_subdirectory(multi-stream-object-detector/cpp multi-stream-object-detector)
