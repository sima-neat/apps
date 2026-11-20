cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rfdetr-detection-segmentation LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

sima_neat_apps_module(
  "rfdetr-detection-segmentation"
  OUTPUT_TARGET_VAR RFDETR_TARGET
  UNIT_TEST
  METADATA_E2E_TEST
)

set(RFDETR_UTILS "${CMAKE_CURRENT_LIST_DIR}/utils/postprocess.cpp")
target_sources(${RFDETR_TARGET} PRIVATE ${RFDETR_UTILS})

# The unit test checks the postprocessing in process, so it compiles the utils too.
if (TARGET rfdetr-detection-segmentation_unit_test)
  target_sources(rfdetr-detection-segmentation_unit_test PRIVATE ${RFDETR_UTILS})
endif()

# The e2e test reads each test stream's frame size with OpenCV.
if (TARGET rfdetr-detection-segmentation_e2e_test)
  find_package(OpenCV REQUIRED COMPONENTS core videoio)
  target_link_libraries(rfdetr-detection-segmentation_e2e_test PRIVATE opencv_videoio)
endif()
