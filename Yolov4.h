// #pragma once
#ifndef YOLOV4_H
#define YOLOV4_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <numeric>
#include <fstream>
#include "dirent.h"
#include "NvOnnxParser.h"
#include "logging.h"
#ifdef __linux__
// #include <X11/Xlib.h>
#elif _WIN32
#endif

//#include "common.hpp"

typedef struct _Config
{
    std::string engine_file;
    std::string labels_file;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    float obj_threshold;
    float nms_threshold;
    bool iou_with_distance;
    std::string model;
    std::vector<int> strides;
    std::vector<int> num_anchors;
    std::vector<std::vector<int>> anchors;
} Config;

class YOLOv4
{
public:
    struct DetectRes
    {
        unsigned int classes;
        float x;
        float y;
        float w;
        float h;
        float prob;
    };
    std::map<int, std::string> detect_labels;
    YOLOv4(Config *config);
    ~YOLOv4();
    void LoadEngine();
    std::vector<DetectRes> EngineInference(cv::Mat &image);
    std::vector<cv::Scalar> class_colors;

private:
    void GenerateReferMatrix();
    std::vector<float> prepareImage(cv::Mat &img);
    std::vector<DetectRes> postProcess(cv::Mat &mat, float *output, int &outSize);
    void NmsDetect(std::vector<DetectRes> &detections);
    float IOUCalculate(const DetectRes &det_a, const DetectRes &det_b);
    float sigmoid(float in);

    Logger gLogger{Severity::kINFO};

    std::string onnx_file;
    std::string engine_file;
    std::string labels_file;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    int CATEGORY;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    float obj_threshold;
    float nms_threshold;
    std::string model_name;
    int refer_rows;
    int refer_cols;
    cv::Mat refer_matrix;
    std::vector<int> strides;
    std::vector<int> num_anchors;
    std::vector<std::vector<int>> anchors;
    std::vector<std::vector<int>> grids;
    bool iou_with_distance;

    void *buffers[2];
    cudaStream_t stream;
    int outSize;
    std::vector<int64_t> bufferSize;
};



std::map<int, std::string> readCOCOLabel(const std::string &fileName);

bool readTrtFile(const std::string &engineFile, nvinfer1::ICudaEngine *&engine, Logger &gLogger);

inline unsigned int getElementSize(nvinfer1::DataType t);

inline int64_t volume(const nvinfer1::Dims &d);

#endif // YOLOV4_H
