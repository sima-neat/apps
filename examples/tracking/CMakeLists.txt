foreach(example IN ITEMS
    multi-camera-people-detection-and-tracking-optiview
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/cpp/CMakeLists.txt")
    add_subdirectory(${example}/cpp ${example})
  endif()
endforeach()
