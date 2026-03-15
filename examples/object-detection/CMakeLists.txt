foreach(example IN ITEMS
    single-rtsp-object-detection-optiview
    multistream-rtsp-detection-pipeline
    multi-camera-people-detection-and-tracking
    simple-object-detection-overlay-pipeline
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/cpp/CMakeLists.txt")
    add_subdirectory(${example}/cpp ${example})
  endif()
endforeach()
