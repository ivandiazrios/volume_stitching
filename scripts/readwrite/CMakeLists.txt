# This is the root ITK CMakeLists file.
cmake_minimum_required(VERSION 2.8.9)
if(COMMAND CMAKE_POLICY)
  cmake_policy(SET CMP0003 NEW)
endif()


# This project is designed to be built outside the Insight source tree.
project(readwrite)

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(readwrite readwrite.cxx)
add_executable(fileinfo fileinfo.cxx)

target_link_libraries(readwrite ${ITK_LIBRARIES})
target_link_libraries(fileinfo ${ITK_LIBRARIES})
