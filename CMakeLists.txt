cmake_minimum_required(VERSION 2.8)

project(Individual)

add_subdirectory("libs/niftiio")

option(USE_CUDA "USE_CUDA" OFF)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

file(GLOB_RECURSE STITCHING_SRC "src/*.cpp" "src/*.cu")
file(GLOB_RECURSE STITCHING_HDR "include/*.hpp" "include/*.cuh")

set(STITCHING_INCLUDE_DIRS "include/")

find_package(Boost COMPONENTS system filesystem REQUIRED)

include_directories(${STITCHING_INCLUDE_DIRS})

if(USE_CUDA)
    
	find_package(CUDA)
	include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
    
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-rdc=true;-ftz=true")
	set(CUDA_SEPARABLE_COMPILATION ON)
    
	add_definitions(-DUSE_CUDA)
    
    cuda_add_executable(stitching ${STITCHING_SRC} ${STITCHING_HDR})

else(USE_CUDA)
    
	add_executable(stitching ${STITCHING_SRC} ${STITCHING_HDR})

endif(USE_CUDA)

target_link_libraries(stitching niftiio ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})

install (TARGETS stitching DESTINATION bin)
