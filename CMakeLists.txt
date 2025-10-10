cmake_minimum_required(VERSION 3.16)
project(mqtt_msg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(mqtt_msg
    src/main.cpp
    src/config.cpp
    src/command.cpp
    src/pipeline.cpp
)

target_include_directories(mqtt_msg PRIVATE include)

