cmake_minimum_required(VERSION 3.7)
project(PIXLAB_TINY_DREAM)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funsafe-math-optimizations -Ofast -flto=auto  -funroll-all-loops -pipe -march=native -Wall -Wextra")

find_package(ncnn REQUIRED)
if (ncnn_FOUND)
	message("NCNN inference engine available")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ncnn_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ncnn_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ncnn_EXE_LINKER_FLAGS}")
    message(STATUS "ncnn_LIBS: ${ncnn_LIBS}")
    message(STATUS "ncnn_INCLUDE_DIRS: ${ncnn_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "NCNN inference engine not found. Please install the library first!")
endif ()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ncnn/include/ncnn)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/ncnn/lib)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/)
add_executable(PIXLAB_TINY_DREAM "boilerplate.cpp" "tinydream.hpp")
target_link_libraries(${PROJECT_NAME} ncnn pthread)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "tinydream")




