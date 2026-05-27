foreach(example IN ITEMS
    face-detector
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/cpp/CMakeLists.txt")
    add_subdirectory(${example}/cpp "${example}_cpp")
  endif()
endforeach()
