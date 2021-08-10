cmake_minimum_required(VERSION 3.5)

project(Yolov4_trt)

set(CMAKE_CXX_STANDARD 20)
set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lX11")

option(INFERENCE_DARKNET "INFERENCE_DARKNET" OFF)
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

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

# OpenCV
find_package(OpenCV REQUIRED)

if (INFERENCE_DARKNET)
    add_definitions(-DINFERENCE_DARKNET)
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
endif(INFERENCE_DARKNET)

include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

if (FOUR_CAMS OR SINGLE_CAM)
    if (INFERENCE_DARKNET)
        add_executable(Streamming VideoCapture.cpp)
        target_link_libraries(Streamming ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} -lX11 libdarknet.so)
    else()
        add_executable(Streamming VideoCapture.cpp Yolov4.cpp)
        target_link_libraries(Streamming ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} -lX11 ${TENSORRT_LIBRARY})
    endif(INFERENCE_DARKNET)
elseif (NOBI_CAMERA_AI_API)
    if (INFERENCE_DARKNET)
        add_executable(Nobi_Camera_AI main.cpp)
        target_link_libraries(Nobi_Camera_AI ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} -lX11 libdarknet.so)
    else()
        add_executable(Nobi_Camera_AI main.cpp Yolov4.cpp)
        target_link_libraries(Nobi_Camera_AI ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} -lX11)
    endif(INFERENCE_DARKNET)
else ()
    if (INFERENCE_DARKNET)
        add_executable(Nobi_App main.cpp)
        target_link_libraries(Nobi_App ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} -lX11 libdarknet.so)
    else()
        add_executable(Nobi_App main.cpp Yolov4.cpp)
        target_link_libraries(Nobi_App ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} -lX11)
    endif(INFERENCE_DARKNET)
endif ()


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
# ./Nobi_Camera_AI \
# --engine-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.engine" \
# --label-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names" \
# --save-dir "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/nobi-hw-videocapture/EMoi" \
# --dims 512 512 --obj-thres 0.5 --nms-thres 0.5 --type-yolo csp --dont-show

# [v] cmake -D NOBI_CAMERA_AI_API=ON -D VIDEO_EXAMPLES=ON ..
# [v] cmake -D NOBI_CAMERA_AI_API=ON -D CAM_ID_EXAMPLES=ON ..
# [v] cmake -D NOBI_CAMERA_AI_API=ON -D VIDEO_EXAMPLES=ON -D DEBUG=ON ..
# [v] cmake -D NOBI_CAMERA_AI_API=ON -D CAM_ID_EXAMPLES=ON -D DEBUG=ON ..
# ./Nobi_App \
# --engine-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.engine" \
# --label-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names" \
# --save-dir "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/AlphaPose/nobi-hw-videocapture/EMoi///" \
# --dims 512 512 --obj-thres 0.3 --nms-thres 0.3 --type-yolo csp --dont-show

# [v] cmake -DINFERENCE_DARKNET=ON -DINFERENCE_VIDEO=ON -DVIDEO_EXAMPLES=ON ..
# [v] cmake -DINFERENCE_DARKNET=ON -DTENSORRT_API=ON -DVIDEO_EXAMPLES=ON -DDEBUG=ON ..
# ./Nobi_App \
# --weights-file /mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/yolov4-csp/yolov4-csp-512.weights \
# --cfg-file /mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/yolov4-csp/yolov4-csp-512.cfg \
# --names-file /mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/yolov4-csp/yolov4-csp-512.names \
# --thresh 0.5 --dont-show

