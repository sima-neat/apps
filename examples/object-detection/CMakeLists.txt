foreach(example IN ITEMS
    high-density-multi-stream-object-detector
    pcie-high-density-multi-stream-object-detector
    detr-object-detector
    rfdetr-detection-segmentation
    single-stream-object-detector
    ssd-mobilenet-object-detector
    yolo26-object-detector
)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/src/cpp/CMakeLists.txt")
    add_subdirectory(${example}/src/cpp ${example})
  endif()
endforeach()
add_subdirectory(multi-stream-object-detector/src/cpp multi-stream-object-detector)
