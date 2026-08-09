cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(multi-stream-people-tracker LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/ExampleModule.cmake")

find_package(PkgConfig REQUIRED)
pkg_check_modules(TRACKER_OPENCV REQUIRED opencv4)

# Some Modalix SDK layouts contain both OpenCV archives and DSOs while selected
# unversioned shared-library links are absent.  Prefer the versioned shared
# objects for the camera-motion path so a transitive -lopencv_* cannot mix a
# static component with OpenCV DSOs in the same link.
set(TRACKER_OPENCV_LINK_LIBRARIES)
set(TRACKER_OPENCV_ALL_SHARED TRUE)
foreach(component IN ITEMS video calib3d features2d imgcodecs imgproc core)
  string(TOUPPER "${component}" component_upper)
  find_library(TRACKER_OPENCV_${component_upper}_SHARED
    NAMES "libopencv_${component}.so.${TRACKER_OPENCV_VERSION}"
    PATHS ${TRACKER_OPENCV_LIBRARY_DIRS}
    NO_DEFAULT_PATH
  )
  if (NOT TRACKER_OPENCV_${component_upper}_SHARED)
    set(TRACKER_OPENCV_ALL_SHARED FALSE)
  endif()
endforeach()

if (TRACKER_OPENCV_ALL_SHARED)
  foreach(component IN ITEMS video calib3d features2d imgcodecs imgproc core)
    string(TOUPPER "${component}" component_upper)
    list(APPEND TRACKER_OPENCV_LINK_LIBRARIES
      "${TRACKER_OPENCV_${component_upper}_SHARED}"
    )
  endforeach()
else()
  # Some SDK releases intentionally ship features2d only as an archive.  In
  # that layout, linking the archive after a shared calib3d is invalid because
  # the DSO cannot bind features2d's hidden C++ symbols.  Keep that dependency
  # pair and its flann dependency consistently static, then use the independent
  # imgcodecs/imgproc/core DSOs.  This avoids the larger all-static OpenCV
  # closure.
  foreach(component IN ITEMS calib3d features2d flann)
    string(TOUPPER "${component}" component_upper)
    find_library(TRACKER_OPENCV_${component_upper}_STATIC
      NAMES "libopencv_${component}.a"
      PATHS ${TRACKER_OPENCV_LIBRARY_DIRS}
      NO_DEFAULT_PATH
    )
    if (NOT TRACKER_OPENCV_${component_upper}_STATIC)
      message(FATAL_ERROR
        "The tracker requires the OpenCV ${component} archive when shared "
        "camera-motion components are incomplete."
      )
    endif()
    list(APPEND TRACKER_OPENCV_LINK_LIBRARIES
      "${TRACKER_OPENCV_${component_upper}_STATIC}"
    )
  endforeach()
  foreach(component IN ITEMS video imgcodecs imgproc core)
    string(TOUPPER "${component}" component_upper)
    if (NOT TRACKER_OPENCV_${component_upper}_SHARED)
      message(FATAL_ERROR
        "The tracker requires the shared OpenCV ${component} component."
      )
    endif()
    list(APPEND TRACKER_OPENCV_LINK_LIBRARIES
      "${TRACKER_OPENCV_${component_upper}_SHARED}"
    )
  endforeach()
endif()

get_filename_component(APPS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
if (NOT TARGET SimaNeatApps::sima_neat)
  _sima_neat_apps_ensure_neat_target("${APPS_ROOT}")
endif()
if (NOT TARGET SimaNeatApps::support_runtime)
  _sima_neat_apps_ensure_support_runtime("${APPS_ROOT}")
endif()

add_library(multi_stream_people_tracker_lib STATIC
  "${CMAKE_CURRENT_LIST_DIR}/utils/camera_motion.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/utils/tracker.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/utils/tracker_overlay.cpp"
)
target_include_directories(multi_stream_people_tracker_lib
  PUBLIC
    "${APPS_ROOT}"
    ${TRACKER_OPENCV_INCLUDE_DIRS}
)
target_compile_definitions(multi_stream_people_tracker_lib
  PUBLIC
    MULTI_CAMERA_PEOPLE_TRACKING_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}"
)

sima_neat_apps_module(
  "multi-stream-people-tracker"
  UNIT_TEST
  E2E_TEST
  OUTPUT_TARGET_VAR MULTI_CAMERA_PEOPLE_TRACKING_TARGET
  SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
)

# Put the tracker archive and its shared OpenCV dependencies before Core's
# transitive, unversioned OpenCV flags.  Link order matters when an SDK contains
# both archives and DSOs but omits selected unversioned shared-library links.
function(link_multi_stream_people_tracker target)
  get_target_property(existing_link_libraries "${target}" LINK_LIBRARIES)
  if (NOT existing_link_libraries)
    set(existing_link_libraries)
  endif()
  set_property(TARGET "${target}" PROPERTY LINK_LIBRARIES
    multi_stream_people_tracker_lib
    ${TRACKER_OPENCV_LINK_LIBRARIES}
    ${existing_link_libraries}
  )
endfunction()

link_multi_stream_people_tracker("${MULTI_CAMERA_PEOPLE_TRACKING_TARGET}")

# Keep a production-faithful tracker-only replay executable next to the app.
# Unlike the Python reference replay, this target exercises the exact C++
# association and covariance implementation deployed on Modalix.
add_executable(multi-stream-people-tracker_replay
  "${CMAKE_CURRENT_LIST_DIR}/replay_tracking.cpp"
)
target_link_libraries(multi-stream-people-tracker_replay
  PRIVATE
    nlohmann_json::nlohmann_json
)
link_multi_stream_people_tracker(multi-stream-people-tracker_replay)

if (TARGET multi-stream-people-tracker_unit_test)
  link_multi_stream_people_tracker(multi-stream-people-tracker_unit_test)
endif()
