foreach(example IN ITEMS
    retinaface-face-detection
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/cpp/CMakeLists.txt")
    add_subdirectory(${example}/cpp "${example}_cpp")
  endif()
endforeach()
