cmake_minimum_required(VERSION 3.15)
project(mqtt_msg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(json INTERFACE)
target_include_directories(json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES
    src/main.cpp
    src/config.cpp
    src/command.cpp
    src/scenario.cpp
    src/model.cpp
    src/pipeline.cpp
    src/rtsp.cpp
    src/mqtt.cpp
)

add_executable(mqtt_msg ${SOURCES})

target_include_directories(mqtt_msg PRIVATE include)

target_link_libraries(mqtt_msg PRIVATE json)

