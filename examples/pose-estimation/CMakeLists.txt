foreach(example IN ITEMS
    yolo26-pose-estimator
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/src/cpp/CMakeLists.txt")
    add_subdirectory(${example}/src/cpp ${example})
  endif()
endforeach()
