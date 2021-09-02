#include "main.h"
#ifdef INFERENCE_ALPHAPOSE_TORCH
#define pose_box bbox
#endif // INFERENCE_ALPHAPOSE_TORCH

// ./Yolov4_trt --engine-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.engine" --label-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names" --dims 512 512 --obj-thres 0.3 --nms-thres 0.3 --type-yolo csp --dont-show
int main(int argc, char **argv)
{
#ifdef INFERENCE_ALPHAPOSE_TORCH
    std::string alphapose_model;
#ifdef INFERENCE_TABULAR_TORCH
    std::string tabular_model;
#endif // INFERENCE_TABULAR_TORCH
#endif // INFERENCE_ALPHAPOSE_TORCH
#ifdef INFERENCE_DARKNET
    std::string weights_file;
    std::string cfg_file;
    std::string names_file;
    float thresh = 0.5;
    bool dont_show = false;
    ParseCommandLine(argc, argv, weights_file, names_file, cfg_file
#ifdef INFERENCE_ALPHAPOSE_TORCH
                     ,
                     alphapose_model
#endif // INFERENCE_ALPHAPOSE_TORCH
                     ,
                     thresh, dont_show);
    Detector yolo(cfg_file, weights_file);
    std::vector<cv::String> obj_names = objects_names_from_file(names_file);
    int const colors[6][3] = {{1, 0, 1}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}};
#else
    Config *cfg = new Config;
    cfg->BATCH_SIZE = 1;
    cfg->INPUT_CHANNEL = 3;
    cfg->strides = std::vector<int>{8, 16, 32};
    cfg->num_anchors = std::vector<int>{3, 3, 3};
    cfg->anchors = std::vector<std::vector<int>>{{12, 16}, {19, 36}, {40, 28}, {36, 75}, {76, 55}, {72, 146}, {142, 110}, {192, 243}, {459, 401}};
    cfg->iou_with_distance = false;
    bool dont_show = false;
    ParseCommandLine(argc, argv, cfg, dont_show
#ifdef INFERENCE_ALPHAPOSE_TORCH
                     ,
                     alphapose_model
#ifdef INFERENCE_TABULAR_TORCH
                     ,
                     tabular_model
#endif // INFERENCE_TABULAR_TORCH
#endif // INFERENCE_ALPHAPOSE_TORCH
    );

    YOLOv4 *yolo = new YOLOv4(cfg);
    yolo->LoadEngine();
#endif // INFERENCE_DARKNET
#ifdef INFERENCE_ALPHAPOSE_TORCH
    AlphaPose *al = new AlphaPose(alphapose_model);
#ifdef INFERENCE_TABULAR_TORCH && !INFERENCE_DARKNET
    Tabular *tab = new Tabular(tabular_model);
#endif // INFERENCE_TABULAR_TORCH && !INFERENCE_DARKNET
#endif // INFERENCE_ALPHAPOSE_TORCH
    while (1)
    {
        std::string image_path;
        std::cout << "Enter Image Path: " << std::endl;
        std::cout.flush();
        std::cin >> image_path;
        cv::Mat image = cv::imread(image_path);
#ifdef INFERENCE_DARKNET
        std::vector<bbox_t> result = yolo.detect_cv(image, thresh, true);
#else
        std::vector<YOLOv4::DetectRes> result = yolo->EngineInference(image);
#endif // INFERENCE_DARKNET
#ifdef INFERENCE_ALPHAPOSE_TORCH
        std::vector<pose_box> inputPose;
#endif // INFERENCE_ALPHAPOSE_TORCH
#ifdef INFERENCE_DARKNET
#ifdef DEBUG
        std::cout << "result.size\t" << result.size() << std::endl;
#endif // DEBUG
        for (const bbox_t &rect : result)
        {
            int x_left = (rect.x < 0) ? 0 : rect.x;
            int y_top = (rect.y < 0) ? 0 : rect.y;
#ifndef JSON
            std::cout << obj_names[rect.obj_id] << ": " << static_cast<int>(rect.prob * 100)
                      << "%\tx_left:  " << static_cast<int>(x_left)
                      << "   y_top:  " << static_cast<int>(y_top)
                      << "   width:  " << static_cast<int>(rect.w)
                      << "   height:  " << static_cast<int>(rect.h)
                      << std::endl;
            std::cout.flush();
#endif // !JSON

            // if (!dont_show)
            {
                char t[256];
                sprintf(t, "%.2f", rect.prob);
                std::string name = obj_names[rect.obj_id] + "-" + t;
                cv::putText(image, name, cv::Point(x_left, y_top - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, obj_id_to_color(rect.obj_id), 2);
                cv::Rect rst(x_left, y_top, rect.w, rect.h);
                cv::rectangle(image, rst, obj_id_to_color(rect.obj_id), 2, cv::LINE_8, 0);
            }

#ifdef INFERENCE_ALPHAPOSE_TORCH
            if (rect.obj_id > 4)
                continue;
            else
            {
                pose_box b;
                M::convert_DetectRes_bbox(rect, b);
                inputPose.push_back(b);
            }
#endif // INFERENCE_ALPHAPOSE_TORCH
        }
#else
        for (const YOLOv4::DetectRes &rect : result)
        {
            int x_left = rect.x - rect.w / 2;
            x_left = (x_left < 0) ? 0 : x_left;
            int y_top = rect.y - rect.h / 2;
            y_top = (y_top < 0) ? 0 : y_top;
#ifndef JSON
            std::cout << yolo->detect_labels[rect.classes] << ": " << static_cast<int>(rect.prob * 100)
                      << "%\tx_left:  " << static_cast<int>(x_left)
                      << "   y_top:  " << static_cast<int>(y_top)
                      << "   width:  " << static_cast<int>(rect.w)
                      << "   height:  " << static_cast<int>(rect.h)
                      << std::endl;
            std::cout.flush();
#endif // !JSON \
       // if (!dont_show)
            {
                char t[256];
                sprintf(t, "%.2f", rect.prob);
                std::string name = yolo->detect_labels[rect.classes] + "-" + t;
                cv::putText(image, name, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, yolo->class_colors[rect.classes], 2);
                cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
                cv::rectangle(image, rst, yolo->class_colors[rect.classes], 2, cv::LINE_8, 0);
            }
#ifdef INFERENCE_ALPHAPOSE_TORCH
            if (rect.classes > 4)
                continue;
            else
            {
                pose_box b;
                M::convert_DetectRes_bbox(rect, b);
                inputPose.push_back(b);
            }
#endif // INFERENCE_ALPHAPOSE_TORCH
        }
#endif // INFERENCE_DARKNET
#ifdef INFERENCE_ALPHAPOSE_TORCH
        std::vector<PoseKeypoints> pKps;
        al->predict(image, inputPose, pKps);
        al->draw(image, pKps);
#ifdef INFERENCE_TABULAR_TORCH
        std::vector<int> tabular_pred;
        tab->predict(inputPose, pKps, tabular_pred);
#endif // INFERENCE_TABULAR_TORCH
#endif // INFERENCE_ALPHAPOSE_TORCH
#ifdef JSON
        std::cout << "[JSON] " << M::res_to_json(result
#ifdef INFERENCE_ALPHAPOSE_TORCH
                                                 ,
                                                 pKps
#ifdef INFERENCE_TABULAR_TORCH
                                                 ,
                                                 tabular_pred
#endif // INFERENCE_TABULAR_TORCH
#endif // INFERENCE_ALPHAPOSE_TORCH
                                                 )
                                      .dump(3)
                  << std::endl;
        std::cout.flush();
#endif // JSON
        if (!dont_show)
        {
            cv::imshow("HNIW", image);
            char c = (char)cv::waitKey(0);
        }
        else
        {
            cv::imwrite(gen_uuid("./", ".jpg"), image);
            cv::imwrite("result.jpg", image);
        }
        image.release();
    }
    return 0;
}

