cmake_minimum_required(VERSION 3.16)
project(InspectAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(-D_DEBUG_)

add_executable(InspectAI
    src/main.cpp
    src/config.cpp
    src/command.cpp
    src/pipeline.cpp
    src/rtsp.cpp
    src/cnn.cpp
    src/yolo.cpp
    src/mqtt.cpp
    src/model.cpp
    src/scenario.cpp
	src/common.cpp
)

target_include_directories(InspectAI PRIVATE include)

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
    target_compile_definitions(InspectAI PRIVATE APP_HAS_ONNXRUNTIME)
    target_include_directories(InspectAI PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(InspectAI PRIVATE ${ONNXRUNTIME_LIBRARY})
endif()

find_path(RKNN_INCLUDE_DIR
    NAMES rknn_api.h
    HINTS $ENV{RKNN_ROOT} $ENV{RKNN_DIR} $ENV{RKNN_SDK_PATH}
    PATH_SUFFIXES include)

find_library(RKNN_LIBRARY
    NAMES rknn_api rknn2
    HINTS $ENV{RKNN_ROOT} $ENV{RKNN_DIR} $ENV{RKNN_SDK_PATH}
    PATH_SUFFIXES lib lib64)

if (RKNN_INCLUDE_DIR AND RKNN_LIBRARY)
    target_compile_definitions(InspectAI PRIVATE APP_HAS_RKNN)
    target_include_directories(InspectAI PRIVATE ${RKNN_INCLUDE_DIR})
    target_link_libraries(InspectAI PRIVATE ${RKNN_LIBRARY})
endif()

find_path(RKMPP_INCLUDE_DIR
    NAMES rk_mpi.h
    HINTS $ENV{RKMPP_ROOT} $ENV{ROCKCHIP_MPP_ROOT}
    PATH_SUFFIXES include)

find_library(RKMPP_LIBRARY
    NAMES rockchip_mpp mpp
    HINTS $ENV{RKMPP_ROOT} $ENV{ROCKCHIP_MPP_ROOT}
    PATH_SUFFIXES lib lib64)

if (RKMPP_INCLUDE_DIR AND RKMPP_LIBRARY)
    target_compile_definitions(InspectAI PRIVATE APP_HAS_RKMPP)
    target_include_directories(InspectAI PRIVATE ${RKMPP_INCLUDE_DIR})
    target_link_libraries(InspectAI PRIVATE ${RKMPP_LIBRARY})
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
target_link_libraries(InspectAI PRIVATE Threads::Threads)

find_package(OpenCV REQUIRED PATHS /usr/include/opencv4)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(InspectAI PRIVATE ${OpenCV_LIBS})

if (MOSQUITTO_FOUND)
    target_compile_definitions(InspectAI PRIVATE APP_HAS_MOSQUITTO)
    target_include_directories(InspectAI PRIVATE ${MOSQUITTO_INCLUDE_DIRS})
    target_link_libraries(InspectAI PRIVATE ${MOSQUITTO_LIBRARIES})
endif()

