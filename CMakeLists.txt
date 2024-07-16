cmake_minimum_required(VERSION 3.12)
project("vision.cpp")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(ggml)

set(VISION_LIB vision.cpp)

file(GLOB VISION_LIB_SOURCES 
    "*.cpp"
    "*.hpp"
)

list(REMOVE_ITEM VISION_LIB_SOURCES "cli.cpp")

add_library(vision ${VISION_LIB_SOURCES})

target_link_libraries(vision PUBLIC ggml)

option(BUILD_CLI "Build CLI application" OFF)

if(BUILD_CLI)
    add_executable(${VISION_LIB} cli.cpp)
    target_link_libraries(${VISION_LIB} PRIVATE vision)
    # turbojpeg.h
    find_library(TURBOJPEG_LIBRARY turbojpeg)
    target_link_libraries(${VISION_LIB} PRIVATE ${TURBOJPEG_LIBRARY})
endif()