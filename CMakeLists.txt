project(simage)

cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

set(SRC "src")
set(CPP ${SRC}/main.cpp)

include_directories(${SRC})
add_executable(simage ${CPP})
