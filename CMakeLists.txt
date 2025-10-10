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
    src/yolo.cpp
    src/mqtt.cpp
)

target_include_directories(mqtt_msg PRIVATE include)

set(ONNXRUNTIME_HINT_DIRS
    $ENV{ONNXRUNTIME_ROOT}
    $ENV{ONNXRUNTIME_DIR}
)

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_HINT_DIRS}
    PATH_SUFFIXES include include/onnxruntime
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_HINT_DIRS}
    PATH_SUFFIXES lib lib64
)

if (ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    target_compile_definitions(mqtt_msg PRIVATE APP_HAS_ONNXRUNTIME)
    target_include_directories(mqtt_msg PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(mqtt_msg PRIVATE ${ONNXRUNTIME_LIBRARY})
endif()

find_package(PkgConfig QUIET)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(MOSQUITTO QUIET libmosquitto)
endif()

if (NOT MOSQUITTO_FOUND)
    find_path(MOSQUITTO_INCLUDE_DIR
        NAMES mosquitto.h
        HINTS $ENV{MOSQUITTO_ROOT} $ENV{MOSQUITTO_DIR}
        PATH_SUFFIXES include)
    find_library(MOSQUITTO_LIBRARY
        NAMES mosquitto
        HINTS $ENV{MOSQUITTO_ROOT} $ENV{MOSQUITTO_DIR}
        PATH_SUFFIXES lib lib64)
    if (MOSQUITTO_INCLUDE_DIR AND MOSQUITTO_LIBRARY)
        set(MOSQUITTO_FOUND ON)
        set(MOSQUITTO_INCLUDE_DIRS ${MOSQUITTO_INCLUDE_DIR})
        set(MOSQUITTO_LIBRARIES ${MOSQUITTO_LIBRARY})
    endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(mqtt_msg PRIVATE Threads::Threads)

if (MOSQUITTO_FOUND)
    target_compile_definitions(mqtt_msg PRIVATE APP_HAS_MOSQUITTO)
    target_include_directories(mqtt_msg PRIVATE ${MOSQUITTO_INCLUDE_DIRS})
    target_link_libraries(mqtt_msg PRIVATE ${MOSQUITTO_LIBRARIES})
endif()

