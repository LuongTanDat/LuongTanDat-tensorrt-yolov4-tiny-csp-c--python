cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(Yolov4_trt)

set(CMAKE_CXX_STANDARD 20)
set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lboost_system -lboost_thread")
set(APP Nobi_App)
# set(CMAKE_PREFIX_PATH /usr/local/libtorch)

option(INFERENCE_DARKNET "INFERENCE_DARKNET" OFF)
option(INFERENCE_ALPHAPOSE_TORCH "INFERENCE_ALPHAPOSE_TORCH" OFF)
option(INFERENCE_VIDEO "Inference videos in realtime" OFF)
option(TENSORRT_API "TensorRT API binding python" OFF)
option(NOBI_CAMERA_AI_API "TensorRT integrate Nobi's Camera AI" OFF)
option(DEBUG "Logging" OFF)
option(FOUR_CAMS "capture from four sources" OFF)
option(SINGLE_CAM "capture from single source" OFF)
option(CAM_ID_EXAMPLES "Open cam 0 1 2 3" OFF)
option(VIDEO_EXAMPLES "Open example video 0 1 2 3" OFF)

if (INFERENCE_VIDEO)
    add_definitions(-DINFERENCE_VIDEO)
endif (INFERENCE_VIDEO)

if (TENSORRT_API)
    add_definitions(-DTENSORRT_API)
endif (TENSORRT_API)

if (NOBI_CAMERA_AI_API)
    add_definitions(-DNOBI_CAMERA_AI_API)
endif (NOBI_CAMERA_AI_API)

if (DEBUG)
    add_definitions(-DDEBUG)
endif (DEBUG)

if (FOUR_CAMS)
    add_definitions(-DFOUR_CAMS)
endif (FOUR_CAMS)

if (SINGLE_CAM)
    add_definitions(-DSINGLE_CAM)
endif (SINGLE_CAM)

if (CAM_ID_EXAMPLES)
    add_definitions(-DCAM_ID_EXAMPLES)
endif (CAM_ID_EXAMPLES)

if (VIDEO_EXAMPLES)
    add_definitions(-DVIDEO_EXAMPLES)
endif (VIDEO_EXAMPLES)

set(LIB ${LIB} -lboost_system -lboost_thread)

# CUDA
find_package(CUDA REQUIRED)
set(LIB ${LIB} ${CUDA_LIBRARIES})
set(INC ${INC} ${CUDA_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
set(LIB ${LIB} ${OpenCV_LIBRARIES})
set(INC ${INC} ${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED)

if (FOUR_CAMS OR SINGLE_CAM)
    set(SRC ${SRC} VideoCapture.cpp)
else()
    set(SRC ${SRC} main.cpp)
endif(FOUR_CAMS OR SINGLE_CAM)

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

# [v] cmake -D SINGLE_CAM=ON -D VIDEO_EXAMPLES=ON ..
# [v] cmake -D SINGLE_CAM=ON -D CAM_ID_EXAMPLES=ON ..
# [v] cmake -D FOUR_CAMS=ON -D VIDEO_EXAMPLES=ON ..
# [v] cmake -D FOUR_CAMS=ON -D CAM_ID_EXAMPLES=ON ..

# [v] cmake -D INFERENCE_VIDEO=ON -D VIDEO_EXAMPLES=ON ..
# [v] cmake -D INFERENCE_VIDEO=ON -D CAM_ID_EXAMPLES=ON ..
# [v] cmake -D INFERENCE_VIDEO=ON -D VIDEO_EXAMPLES=ON -D DEBUG=ON ..
# [v] cmake -D INFERENCE_VIDEO=ON -D CAM_ID_EXAMPLES=ON -D DEBUG=ON ..

# [v] cmake -D TENSORRT_API=ON ..
# ./Nobi_Trt \
# --engine-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.engine" \
# --label-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names" \
# --dims 512 512 --obj-thres 0.3 --nms-thres 0.3 --type-yolo csp --dont-show

# [v] cmake -D NOBI_CAMERA_AI_API=ON -D VIDEO_EXAMPLES=ON ..
# [v] cmake -D NOBI_CAMERA_AI_API=ON -D CAM_ID_EXAMPLES=ON ..
# [v] cmake -D NOBI_CAMERA_AI_API=ON -D VIDEO_EXAMPLES=ON -D DEBUG=ON ..
# [v] cmake -D NOBI_CAMERA_AI_API=ON -D CAM_ID_EXAMPLES=ON -D DEBUG=ON ..
# ./${APP} \
# --engine-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.engine" \
# --label-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names" \
# --save-dir "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/nobi-hw-videocapture/EMoi" \
# --dims 512 512 --obj-thres 0.5 --nms-thres 0.5 --type-yolo csp --dont-show

# [v] cmake -D NOBI_CAMERA_AI_API=ON -D VIDEO_EXAMPLES=ON ..
# [v] cmake -D NOBI_CAMERA_AI_API=ON -D CAM_ID_EXAMPLES=ON ..
# [v] cmake -D NOBI_CAMERA_AI_API=ON -D VIDEO_EXAMPLES=ON -D DEBUG=ON ..
# [v] cmake -D NOBI_CAMERA_AI_API=ON -D CAM_ID_EXAMPLES=ON -D DEBUG=ON ..
# ./${APP} \
# --engine-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.engine" \
# --label-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names" \
# --save-dir "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/AlphaPose/nobi-hw-videocapture/EMoi///" \
# --dims 512 512 --obj-thres 0.3 --nms-thres 0.3 --type-yolo csp --dont-show

# [v] cmake -DINFERENCE_DARKNET=ON -DINFERENCE_VIDEO=ON -DVIDEO_EXAMPLES=ON ..
# [v] cmake -DINFERENCE_DARKNET=ON -DTENSORRT_API=ON -DVIDEO_EXAMPLES=ON -DDEBUG=ON ..
# ./${APP} \
# --weights-file /mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/yolov4-csp/yolov4-csp-512.weights \
# --cfg-file /mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/yolov4-csp/yolov4-csp-512.cfg \
# --names-file /mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/yolov4-csp/yolov4-csp-512.names \
# --thresh 0.5 --dont-show
# [ ] cmake -DINFERENCE_VIDEO=ON -DVIDEO_EXAMPLES=ON -DINFERENCE_ALPHAPOSE_TORCH=ON ..
