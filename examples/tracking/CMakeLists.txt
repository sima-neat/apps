foreach(example IN ITEMS
    multi-stream-people-tracker
    yolo26-tiny-drone-tracker
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/src/cpp/CMakeLists.txt")
    add_subdirectory(${example}/src/cpp ${example})
  endif()
endforeach()
