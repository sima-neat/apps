foreach(example IN ITEMS
    single-rtsp-object-detection-insight
    simple-object-detection-overlay-pipeline
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/cpp/CMakeLists.txt")
    add_subdirectory(${example}/cpp ${example})
  endif()
endforeach()
add_subdirectory(multistream-object-detection-insight/cpp multistream-object-detection-insight)
