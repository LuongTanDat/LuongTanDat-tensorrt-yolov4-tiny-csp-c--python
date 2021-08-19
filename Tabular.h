#ifndef TABULAR_H
#define TABULAR_H
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <map>
#include <opencv2/opencv.hpp>
#include <assert.h>
#include "AlphaPose.h"
#define CATEGORY 7
#define NUM_KEYPOINT 17

class Tabular
{
public:
    Tabular(std::string model_path);
    void predict(std::vector<bbox> &objBoxes, std::vector<PoseKeypoints> &poseKeypoints, std::vector<int> &tabular_predict);

private:
    torch::jit::script::Module tab;
    void preprocess(torch::Tensor &input_data, std::vector<bbox> &cropped_boxes, std::vector<PoseKeypoints> &poseKeypoints);
};
#endif // TABULAR_H