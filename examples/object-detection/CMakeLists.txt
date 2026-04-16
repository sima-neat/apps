foreach(example IN ITEMS
    detr-object-detection
    single-rtsp-object-detection-optiview
    multistream-rtsp-detection-pipeline
    simple-object-detection-overlay-pipeline
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/cpp/CMakeLists.txt")
    add_subdirectory(${example}/cpp ${example})
  endif()
endforeach()
add_subdirectory(multistream-object-detection-optiview/cpp multistream-object-detection-optiview)
