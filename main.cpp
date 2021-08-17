#include "main.h"
#include "glob.h"
#define pose_box bbox

// ./Yolov4_trt --engine-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.engine" --label-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names" --dims 512 512 --obj-thres 0.3 --nms-thres 0.3 --type-yolo csp --dont-show
int main(int argc, char **argv)
{
    Config *cfg = new Config;
    cfg->BATCH_SIZE = 1;
    cfg->INPUT_CHANNEL = 3;
    cfg->strides = std::vector<int>{8, 16, 32};
    cfg->num_anchors = std::vector<int>{3, 3, 3};
    cfg->anchors = std::vector<std::vector<int>>{{12, 16}, {19, 36}, {40, 28}, {36, 75}, {76, 55}, {72, 146}, {142, 110}, {192, 243}, {459, 401}};
    cfg->iou_with_distance = false;
    cfg->engine_file = "/mnt/642C9F7E0555E58A/Nobi/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.engine";
    cfg->labels_file = "/mnt/642C9F7E0555E58A/Nobi/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names";
    cfg->IMAGE_WIDTH = 512;
    cfg->IMAGE_HEIGHT = 512;
    cfg->obj_threshold = 0.7;
    cfg->nms_threshold = 0.7;
    cfg->model = "csp";
    bool dont_show = false;
    std::string saved_dir = "/mnt/642C9F7E0555E58A/Nobi/nobi-hw-videocapture/EMoi/";
    std::string alphapose_model = "/mnt/642C9F7E0555E58A/Nobi/model-zoo/fast_pose_res50/fast_res50_256x192.jit";

    YOLOv4 *yolo = new YOLOv4(cfg);
    yolo->LoadEngine();
    AlphaPose *al = new AlphaPose(alphapose_model);
    size_t count = 0;
    std::vector<std::string> search_parttern = {"/mnt/642C9F7E0555E58A/Nobi/annotate_valid/", "/mnt/642C9F7E0555E58A/Nobi/lying_sitting_on_bed/"};
    for (std::string parttern : search_parttern)
    {
        glob::glob glob(parttern + "*.jpg");
        while (glob)
        {
            std::string image_path = glob.current_match();
            cv::Mat image = cv::imread(parttern + image_path);
            // std::cout << image.rows << "\t" << image.cols << "\t" << image.channels() << std::endl;
            std::vector<YOLOv4::DetectRes> result = yolo->EngineInference(image);
            std::vector<pose_box> inputPose;
            for (const YOLOv4::DetectRes &rect : result)
            {
                int x_left = rect.x - rect.w / 2;
                x_left = (x_left < 0) ? 0 : x_left;
                int y_top = rect.y - rect.h / 2;
                y_top = (y_top < 0) ? 0 : y_top;

                // if (!dont_show)
                {
                    char t[256];
                    sprintf(t, "%.2f", rect.prob);
                    std::string name = yolo->detect_labels[rect.classes] + "-" + t;
                    cv::putText(image, name, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, yolo->class_colors[rect.classes], 2);
                    cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
                    cv::rectangle(image, rst, yolo->class_colors[rect.classes], 2, cv::LINE_8, 0);
                }
                if (rect.classes > 4)
                    continue;
                else
                {
                    pose_box b;
                    M::convert_DetectRes_bbox(rect, b);
                    inputPose.push_back(b);
                }
            }
            std::vector<PoseKeypoints> pKps;
            al->predict(image, inputPose, pKps);
            al->draw(image, pKps);
            nlohmann::json j = M::res_to_json(result, pKps);
            std::ofstream file;
            std::string saved_file = saved_dir + image_path;
            cv::imwrite(saved_file, image);
            mreplace(saved_file, ".jpg", ".json");
            file.open(saved_file);
            file << j.dump(2);
            file.close();
            image.release();
            std::cout << count++ << "\t" << saved_file << std::endl;
            glob.next();
        }
    }
    return 0;
}
