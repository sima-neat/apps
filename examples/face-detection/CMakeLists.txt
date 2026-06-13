foreach(example IN ITEMS
    face-detector
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/src/cpp/CMakeLists.txt")
    add_subdirectory(${example}/src/cpp "${example}_cpp")
  endif()
endforeach()
