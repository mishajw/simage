set(PROJECT_NAME simage)
set(SRC "src")
set(CPP
        ${SRC}/main.cpp
        ${SRC}/util.cpp
        ${SRC}/util.h
        ${SRC}/image_normalizer.cpp
        ${SRC}/image_normalizer.h
        ${SRC}/image_normalizer_evaluator.cpp
        ${SRC}/image_normalizer_evaluator.h)

project(${PROJECT_NAME})

cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${SRC})
add_executable(${PROJECT_NAME} ${CPP})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
