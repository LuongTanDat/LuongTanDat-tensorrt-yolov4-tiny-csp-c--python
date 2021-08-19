#include "Tabular.h"

Tabular::Tabular(std::string model_path)
{
    try
    {
        std::cout << model_path << std::endl;
        this->tab = torch::jit::load(model_path, torch::kCPU);
    }
    catch (const c10::Error &e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << e.msg() << std::endl;
    }
}

void Tabular::predict(std::vector<bbox> &objBoxes, std::vector<PoseKeypoints> &poseKeypoints, std::vector<int> &tabular_predict)
{
    int batch_size = objBoxes.size();
    torch::Tensor input_data;
    preprocess(input_data, objBoxes, poseKeypoints);
    torch::Tensor output = this->tab.forward({input_data}).toTensor().to(torch::kCPU);
    std::tuple<torch::Tensor, torch::Tensor> mm = torch::max(output.reshape({batch_size, 5}), 1);
    torch::Tensor idx = std::get<1>(mm).to(torch::kInt32);
    
    int *idx_ptr = (int *)idx.data_ptr();
    for (int i = 0; i < objBoxes.size(); i++)
    {
        tabular_predict.push_back((int)(*idx_ptr));
        idx_ptr++;
    }
}

void Tabular::preprocess(torch::Tensor &input_data, std::vector<bbox> &cropped_boxes, std::vector<PoseKeypoints> &poseKeypoints)
{
    assert(cropped_boxes.size() == poseKeypoints.size());
    int batch_size = cropped_boxes.size();
    float batch[batch_size][CATEGORY + 3 * NUM_KEYPOINT];
    for (int i = 0; i < batch_size; i++)
    {
        float *feat = *(batch + i);
        memcpy((void *)(feat + 3 * NUM_KEYPOINT), (void *)(cropped_boxes[i].feat), CATEGORY * sizeof(float));
        for (int j = 0; j < NUM_KEYPOINT; j++)
        {
            *(feat + 3 * j) = (poseKeypoints[i].keypoints[j].x - cropped_boxes[i].rect.x) / cropped_boxes[i].rect.width;
            *(feat + 3 * j + 1) = (poseKeypoints[i].keypoints[j].y - cropped_boxes[i].rect.y) / cropped_boxes[i].rect.height;
            *(feat + 3 * j + 2) = poseKeypoints[i].kp_scores[j];
        }
    }
    input_data = torch::from_blob(
                     (void *)batch,
                     {(long)batch_size, CATEGORY + 3 * NUM_KEYPOINT}, torch::TensorOptions(torch::kFloat))
                     .toType(torch::kFloat32).clone();
}
