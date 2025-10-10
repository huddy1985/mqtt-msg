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
    src/rtsp.cpp
    src/cnn.cpp
)

target_include_directories(mqtt_msg PRIVATE include)

find_package(Torch QUIET)
if (Torch_FOUND)
    target_compile_definitions(mqtt_msg PRIVATE APP_HAS_TORCH)
    target_include_directories(mqtt_msg PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(mqtt_msg PRIVATE ${TORCH_LIBRARIES})
endif()

