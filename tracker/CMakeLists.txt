cmake_minimum_required(VERSION 3.22)

project(tracker)

# Set the C++ standards
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS})

# Create sources
set(SOURCES
    src/main.cpp
)

# Add executable
add_executable(tracker ${SOURCES})
target_link_libraries(tracker ${OpenCV_LIBS} Eigen3::Eigen)