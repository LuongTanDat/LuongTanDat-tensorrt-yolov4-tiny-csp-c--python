cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(Yolov4_trt)

set(CMAKE_CXX_STANDARD 20)
set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lboost_system -lboost_thread")
set(APP Nobi_App)
# set(CMAKE_PREFIX_PATH /usr/local/libtorch)
# set(CMAKE_PREFIX_PATH /usr)
set(PATH "/usr/local/cuda/bin" ${PATH})
set(LD_LIBRARY_PATH "/usr/local/cuda/lib64:/usr/local/TensorRT/lib" ${LD_LIBRARY_PATH})
set(Torch_DIR "/usr/local/lib/python3.6/dist-packages/torch" ${Torch_DIR})

option(INFERENCE_DARKNET "INFERENCE_DARKNET" OFF)
option(INFERENCE_ALPHAPOSE_TORCH "INFERENCE_ALPHAPOSE_TORCH" OFF)
option(INFERENCE_TABULAR_TORCH "INFERENCE_TABULAR_TORCH" OFF)
option(JSON "convert output to json" OFF)

if (JSON)
    add_definitions(-DJSON)
endif (JSON)

set(LIB ${LIB} -lboost_system -lboost_thread)

# cURLpp
find_package(CURL REQUIRED)
set(CURL_LIBRARIES ${CURL_LIBRARIES} /usr/local/lib/libcurlpp.so)
set(CURL_INCLUDE_DIRS ${CURL_INCLUDE_DIRS} /usr/local/include/curlpp/)
message("CURL_LIBRARIES: " ${CURL_LIBRARIES})
message("CURL_INCLUDE_DIRS: " ${CURL_INCLUDE_DIRS})
set(LIB ${LIB} ${CURL_LIBRARIES})
set(INC ${INC} ${CURL_INCLUDE_DIRS})

# CUDA
find_package(CUDA REQUIRED)
set(LIB ${LIB} ${CUDA_LIBRARIES})
set(INC ${INC} ${CUDA_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
set(LIB ${LIB} ${OpenCV_LIBRARIES})
set(INC ${INC} ${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED)
set(SRC ${SRC} main.cpp glob.cpp base64.cpp)

# Torch
if (INFERENCE_ALPHAPOSE_TORCH)
    add_definitions(-DINFERENCE_ALPHAPOSE_TORCH)
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
    message(STATUS "CMAKE_CXX_FLAGS:     ${CMAKE_CXX_FLAGS}")
    message(STATUS "TORCH_INCLUDE_DIRS:  ${TORCH_INCLUDE_DIRS}")
    message(STATUS "TORCH_LIBRARIES:     ${TORCH_LIBRARIES}")
    set(SRC ${SRC} AlphaPose.cpp)
    set(LIB ${LIB} ${TORCH_LIBRARIES})
    set(INC ${INC} ${TORCH_INCLUDE_DIRS})

    if (INFERENCE_TABULAR_TORCH)
        add_definitions(-DINFERENCE_TABULAR_TORCH)
        set(SRC ${SRC} Tabular.cpp)
    endif(INFERENCE_TABULAR_TORCH)
endif(INFERENCE_ALPHAPOSE_TORCH)

if (INFERENCE_DARKNET)
    add_definitions(-DINFERENCE_DARKNET)
    set(LIB ${LIB} libdarknet.so)
else()
    # TensorRT
    set(TENSORRT_ROOT /usr/local/TensorRT)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
    find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
    find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
    set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})
    set(SRC ${SRC} Yolov4.cpp)
    set(LIB ${LIB} ${TENSORRT_LIBRARY})
    set(INC ${INC} ${TENSORRT_INCLUDE_DIR})
endif(INFERENCE_DARKNET)

include_directories(${INC})

add_executable(${APP} ${SRC})
target_link_libraries(${APP} ${LIB})
