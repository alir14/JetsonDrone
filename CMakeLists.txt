# require CMake 2.8 or greater
cmake_minimum_required(VERSION 2.8)

file(GLOB droneprojectSrc *.cpp)
file(GLOB dronprojectInc *.h)

# declare my-recognition project
project(droneProject)

# import jetson-inference and jetson-utils packages.
# note that if you didn't do "sudo make install"
# while building jetson-inference, this will error.
find_package(jetson-utils)
find_package(jetson-inference)

#OPEN CV
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${OpenCV_LIBS})

# CUDA and Qt4 are required
find_package(CUDA)
find_package(Qt4)

# setup Qt4 for build
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

# compile the my-recognition program
cuda_add_executable(${PROJECT_NAME} ${droneprojectSrc})

add_subdirectory(udp)

# link my-recognition to jetson-inference library
target_link_libraries(${PROJECT_NAME} jetson-inference ${OpenCV_LIBS} udpMessagePublisher)
