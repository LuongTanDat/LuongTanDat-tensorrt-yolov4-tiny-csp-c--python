#include "main.h"
#ifdef INFERENCE_ALPHAPOSE_TORCH
#define pose_box bbox
#endif // INFERENCE_ALPHAPOSE_TORCH

#ifdef INFERENCE_VIDEO
#ifdef VIDEO_EXAMPLES
#define TYPE std::string
#define SOURCE0 std::string("filesrc location=/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/samples/DAT_0.mp4 ! decodebin ! autovideoconvert ! appsink")
#define SOURCE1 std::string("filesrc location=/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/samples/DAT_1.mp4 ! decodebin ! autovideoconvert ! appsink")
#define SOURCE2 std::string("filesrc location=/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/samples/DAT_2.mp4 ! decodebin ! autovideoconvert ! appsink")
#define SOURCE3 std::string("filesrc location=/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/samples/DAT_3.mp4 ! decodebin ! autovideoconvert ! appsink")
#define BACKEND cv::CAP_GSTREAMER
#endif // VIDEO_EXAMPLES

#ifdef CAM_ID_EXAMPLES
#define TYPE int
#define SOURCE0 0
#define SOURCE1 1
#define SOURCE2 2
#define SOURCE3 3
#define BACKEND cv::CAP_V4L2
#endif // CAM_ID_EXAMPLES

bool stop_video = false;
M::send_one_replaceable_object<M::pipeline_data> stream2detect;
#ifdef INFERENCE_ALPHAPOSE_TORCH
M::send_one_replaceable_object<M::pipeline_data> detect2pose, pose2show;
#else
M::send_one_replaceable_object<M::pipeline_data> detect2show;
#endif // INFERENCE_ALPHAPOSE_TORCH
std::atomic<int> current_fps_det(0);
std::chrono::steady_clock::time_point fps_count_start;
std::atomic<int> fps_det_counter(0);

int main()
{
#ifdef INFERENCE_DARKNET
    std::string names_file = "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names";
    std::string cfg_file = "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.cfg";
    std::string weights_file = "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.weights";
    float thresh = 0.5;
    Detector yolo(cfg_file, weights_file);
    std::vector<cv::String> obj_names = objects_names_from_file(names_file);
    int const colors[6][3] = {{1, 0, 1}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}};
#else
    Config *cfg = new Config();
    cfg->BATCH_SIZE = 1;
    cfg->INPUT_CHANNEL = 3;
#ifdef YOLOv4_CSP_512
    cfg->engine_file = "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.engine";
    cfg->labels_file = "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names";
    cfg->IMAGE_WIDTH = 512;
    cfg->IMAGE_HEIGHT = 512;
    cfg->model = std::string("csp");
    cfg->obj_threshold = 0.6;
    cfg->nms_threshold = 0.45;
    cfg->strides = std::vector<int>{8, 16, 32};
    cfg->num_anchors = std::vector<int>{3, 3, 3};
    cfg->anchors = std::vector<std::vector<int>>{{12, 16}, {19, 36}, {40, 28}, {36, 75}, {76, 55}, {72, 146}, {142, 110}, {192, 243}, {459, 401}};
#endif // YOLOv4_CSP_512
    YOLOv4 *yolo = new YOLOv4(cfg);
    yolo->LoadEngine();
#endif // INFERENCE_DARKNET

#ifdef INFERENCE_ALPHAPOSE_TORCH
    AlphaPose *al = new AlphaPose("/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/AlphaPose/AlphaPose_TorchScript/model-zoo/fast_pose_res50/fast_res50_256x192.jit");
#endif // INFERENCE_ALPHAPOSE_TORCH

    auto timer_global_start = std::chrono::high_resolution_clock::now();
    std::vector<M::StreamSource<TYPE>> sources;
    sources.push_back(M::StreamSource<TYPE>(SOURCE0, BACKEND, 0, 0, 0));
    sources.push_back(M::StreamSource<TYPE>(SOURCE1, BACKEND, 1, 0, 1));
    sources.push_back(M::StreamSource<TYPE>(SOURCE2, BACKEND, 2, 1, 0));
    sources.push_back(M::StreamSource<TYPE>(SOURCE3, BACKEND, 3, 1, 1));

    M::VideoStream<TYPE> stream(sources);
    stream.set_param(VISUAL_WIDTH, VISUAL_HEIGHT, 2, 2);
    M::pipeline_data final_data;
    cv::Mat merged_frame(cv::Size(VISUAL_WIDTH * 2, VISUAL_HEIGHT * 2), CV_8UC3);
    cv::Mat display_frame;
#ifdef DEBUG
    // int fourcc = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
    int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
    cv::VideoWriter videoWriter;
#endif // DEBUG
    if (!stream.isOpened())
    {
#ifdef DEBUG
        std::cout << "[DEBUG] "
                  << "[API][ERROR] Error opening video stream or file" << std::endl;
#endif // DEBUG
    }
    stop_video = true;

    std::thread retrieve_frame_thead(
        [&]()
        {
            while (stop_video)
            {
                stream.grab_frame();
                cv::Mat merged(cv::Size(VISUAL_WIDTH * 2, VISUAL_HEIGHT * 2), CV_8UC3);
                M::pipeline_data p_data(merged);
                bool state = stream.retrieve_frame(p_data.cap_frame);
                // if (!state)
                //     break;
                stream2detect.send(p_data);
            }
        });

    std::thread detector_thread(
        [&]()
        {
            while (stop_video)
            {
#ifdef DEBUG
                std::cout << "[DEBUG] "
                          << "stop_video\t" << stop_video << std::endl;
#endif // DEBUG
                M::pipeline_data p_data;
                // p_data.result.clear();
                p_data = stream2detect.receive();
                if (p_data.cap_frame.empty())
                {
#ifdef INFERENCE_ALPHAPOSE_TORCH
                    detect2pose.send(p_data);
#else
                    detect2show.send(p_data);
#endif // INFERENCE_ALPHAPOSE_TORCH
                    break;
                }
                p_data.uuid = gen_uuid(std::string("/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/nobi-hw-videocapture/EMoi///"), std::string(".jpg"));
                // cv::imwrite(p_data.uuid, p_data.cap_frame);
                for (int i = 0; i < 4; i++)
                {
                    cv::Mat roi(p_data.cap_frame, cv::Rect((i % 2) * VISUAL_WIDTH, (i / 2) * VISUAL_HEIGHT, VISUAL_WIDTH, VISUAL_HEIGHT));
#ifdef INFERENCE_DARKNET
                    // std::shared_ptr<image_t> det_image = yolo.mat_to_image_resize(roi);
                    // std::vector<bbox_t> result = yolo.detect_resized(*det_image, roi.cols, roi.rows, thresh, true);
                    std::vector<bbox_t> result = yolo.detect_cv(roi, thresh, true);
                    std::pair<int, std::vector<bbox_t>> _pair(i, result);
#else
                    std::vector<YOLOv4::DetectRes> result = yolo->EngineInference(roi);
#ifdef DEBUG
                    std::cout << "[DEBUG] " << p_data.uuid << "\tresult.size\t" << result.size() << "\tcam: " << i << std::endl;
#endif // DEBUG
                    std::pair<int, std::vector<YOLOv4::DetectRes>> _pair(i, result);
#endif // INFERENCE_DARKNET
                    p_data.result.insert(_pair);
                }
#ifdef INFERENCE_ALPHAPOSE_TORCH
                detect2pose.send(p_data);
#else
                detect2show.send(p_data);
#endif // INFERENCE_ALPHAPOSE_TORCH
            }
        });

#ifdef INFERENCE_ALPHAPOSE_TORCH
    std::thread pose_thread(
        [&]()
        {
            while (stop_video)
            {
                M::pipeline_data p_data;
                // p_data.result.clear();
                p_data = detect2pose.receive();
                if (p_data.cap_frame.empty())
                {
                    pose2show.send(p_data);
                    break;
                }
                for (int i = 0; i < 4; i++)
                {
                    cv::Mat roi(p_data.cap_frame, cv::Rect((i % 2) * VISUAL_WIDTH, (i / 2) * VISUAL_HEIGHT, VISUAL_WIDTH, VISUAL_HEIGHT));
                    std::vector<PoseKeypoints> poseKeypoints;
                    std::vector<bbox> objBoxes;
                    M::convert_vecDetectRes_vecbbox(p_data.result[i], objBoxes);
                    al->predict(roi, objBoxes, poseKeypoints);
                    std::pair<int, std::vector<PoseKeypoints>> _pair(i, poseKeypoints);
                    p_data.poseKeypoints.insert(_pair);
                }
                pose2show.send(p_data);
            }
        });
#endif // INFERENCE_ALPHAPOSE_TORCH

    while (stop_video)
    {
#ifdef INFERENCE_ALPHAPOSE_TORCH
        final_data = pose2show.receive();
#else
        final_data = detect2show.receive();
#endif //  INFERENCE_ALPHAPOSE_TORCH
        if (final_data.cap_frame.empty())
            break;
// if (!state)
//     break;
#ifdef INFERENCE_DARKNET
        for (const std::pair<const int, std::vector<bbox_t>> &_pair : final_data.result)
        {
            int cam = _pair.first;
            std::vector<bbox_t> result = _pair.second;
            cv::Mat roi(final_data.cap_frame, cv::Rect((cam % 2) * VISUAL_WIDTH, (cam / 2) * VISUAL_HEIGHT, VISUAL_WIDTH, VISUAL_HEIGHT));

#ifdef DEBUG
            std::cout << "[DEBUG] "
                      << "UUID\t" << final_data.uuid << std::endl;
#endif // DEBUG
            for (const auto &rect : result)
            {
#ifdef DEBUG
                std::ostringstream os;
                os << "cam:" << cam << "  classes:" << obj_names[rect.obj_id] << "  x:" << rect.x << "  y:" << rect.y << "  w:" << rect.w << "  h:" << rect.h << "  prob:" << rect.prob;
                std::cout << "[DEBUG] " << os.str() << std::endl;
#endif // DEBUG
                char t[256];
                sprintf(t, "%.2f", rect.prob);
                std::string name = obj_names[rect.obj_id] + "-" + t;
                cv::putText(roi, name, cv::Point(rect.x, rect.y - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, obj_id_to_color(rect.obj_id), 2);
                cv::Rect rst(rect.x, rect.y, rect.w, rect.h);
                cv::rectangle(roi, rst, obj_id_to_color(rect.obj_id), 2, cv::LINE_8, 0);
            }
        }
#else
        for (const std::pair<const int, std::vector<YOLOv4::DetectRes>> &_pair : final_data.result)
        {
            int cam = _pair.first;
            std::vector<YOLOv4::DetectRes> result = _pair.second;
            cv::Mat roi(final_data.cap_frame, cv::Rect((cam % 2) * VISUAL_WIDTH, (cam / 2) * VISUAL_HEIGHT, VISUAL_WIDTH, VISUAL_HEIGHT));

#ifdef DEBUG
            std::cout << "[DEBUG] "
                      << "UUID\t" << final_data.uuid << std::endl;
#endif // DEBUG
            for (const auto &rect : result)
            {
#ifdef DEBUG
                std::ostringstream os;
                os << "cam:" << cam << "  classes:" << rect.classes << "  x:" << rect.x << "  y:" << rect.y << "  w:" << rect.w << "  h:" << rect.h << "  prob:" << rect.prob;
                std::cout << "[DEBUG] " << os.str() << std::endl;
#endif // DEBUG
                char t[256];
                sprintf(t, "%.2f", rect.prob);
                std::string name = yolo->detect_labels[rect.classes] + "-" + t;
                cv::putText(roi, name, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, yolo->class_colors[rect.classes], 2);
                cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
                cv::rectangle(roi, rst, yolo->class_colors[rect.classes], 2, cv::LINE_8, 0);
            }
        }
#endif

#ifdef INFERENCE_ALPHAPOSE_TORCH
        for (const std::pair<const int, std::vector<PoseKeypoints>> &_pair : final_data.poseKeypoints)
        {
            int cam = _pair.first;
            std::vector<PoseKeypoints> pKp = _pair.second;
            cv::Mat roi(final_data.cap_frame, cv::Rect((cam % 2) * VISUAL_WIDTH, (cam / 2) * VISUAL_HEIGHT, VISUAL_WIDTH, VISUAL_HEIGHT));
            al->draw(roi, pKp);
        }
#endif // INFERENCE_ALPHAPOSE_TORCH

        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        float time_sec = std::chrono::duration<double>(now - fps_count_start).count();

        if (time_sec >= 1)
        {
            current_fps_det = fps_det_counter / time_sec;
            fps_count_start = now;
            fps_det_counter = 0;
        }

        std::string info_msg = " | FPS Detection: " + std::to_string(current_fps_det);

        // If the frame is empty, break immediately
        putText(final_data.cap_frame, info_msg, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
        cv::resize(final_data.cap_frame, display_frame, cv::Size(1600, 900));
        cv::imwrite("result.jpg", display_frame);
#ifdef DEBUG
        if (!videoWriter.isOpened())
        {
            videoWriter.open("result.mp4", fourcc, 10.0, display_frame.size(), true);
            std::cout << "Create result.mp4" << std::endl;
        }

        if (videoWriter.isOpened())
        {
            videoWriter.write(display_frame);
            std::cout << "Write to result.mp4" << std::endl;
        }
#endif // DEBUG
        cv::imwrite("result.jpg", display_frame);
        cv::imshow("final", display_frame);
        // Press  ESC on keyboard to exit
        char c = (char)cv::waitKey(1);
        if (c == 27)
        {
            break;
        }
    }
    stop_video = false;
    if (retrieve_frame_thead.joinable())
        retrieve_frame_thead.join();
    if (detector_thread.joinable())
        detector_thread.join();
#ifdef INFERENCE_ALPHAPOSE_TORCH
    if (pose_thread.joinable())
        pose_thread.join();
#endif // INFERENCE_ALPHAPOSE_TORCH
#ifdef DEBUG
    videoWriter.release();
#endif // DEBUG
    return 0;
}
#endif // INFERENCE_VIDEO
#ifdef TENSORRT_API
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
#endif // TENSORRT_API
#ifdef NOBI_CAMERA_AI_API
// #define CAM_ID_EXAMPLES
// #define VIDEO_EXAMPLES

#ifdef VIDEO_EXAMPLES
#define TYPE std::string
#define SOURCE0 std::string("filesrc location=/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/samples/DAT_0.mp4 ! decodebin ! autovideoconvert ! appsink")
#define SOURCE1 std::string("filesrc location=/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/samples/DAT_1.mp4 ! decodebin ! autovideoconvert ! appsink")
#define SOURCE2 std::string("filesrc location=/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/samples/DAT_2.mp4 ! decodebin ! autovideoconvert ! appsink")
#define SOURCE3 std::string("filesrc location=/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/samples/DAT_3.mp4 ! decodebin ! autovideoconvert ! appsink")
#define BACKEND cv::CAP_GSTREAMER
#endif // VIDEO_EXAMPLES

#ifdef CAM_ID_EXAMPLES
#define TYPE int
#define SOURCE0 0
#define SOURCE1 1
#define SOURCE2 2
#define SOURCE3 3
#define BACKEND cv::CAP_V4L2
#endif // CAM_ID_EXAMPLES

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
    std::string save_dir;
    ParseCommandLine(argc, argv, weights_file, names_file, cfg_file
#ifdef INFERENCE_ALPHAPOSE_TORCH
                     ,
                     alphapose_model
#endif // INFERENCE_ALPHAPOSE_TORCH
                     ,
                     save_dir, thresh, dont_show);
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
    std::string save_dir;
    ParseCommandLine(argc, argv, cfg, dont_show, save_dir
#ifdef INFERENCE_ALPHAPOSE_TORCH
                     ,
                     alphapose_model
#ifdef INFERENCE_TABULAR_TORCH
                     ,
                     tabular_model
#endif // INFERENCE_TABULAR_TORCH
#endif // INFERENCE_ALPHAPOSE_TORCH
    );
#ifdef DEBUG
    std::cout << "[DEBUG] "
              << "save_dir: " << save_dir << std::endl;
#endif // DEBUG

    YOLOv4 *yolo = new YOLOv4(cfg);
    yolo->LoadEngine();
#endif // INFERENCE_DARKNET
#ifdef INFERENCE_ALPHAPOSE_TORCH
    AlphaPose *al = new AlphaPose(alphapose_model);
#ifdef INFERENCE_TABULAR_TORCH && !INFERENCE_DARKNET
    Tabular *tab = new Tabular(tabular_model);
#endif // INFERENCE_TABULAR_TORCH && !INFERENCE_DARKNET
#endif // INFERENCE_ALPHAPOSE_TORCH

    std::vector<M::StreamSource<TYPE>> sources;
    sources.push_back(M::StreamSource<TYPE>(SOURCE0, BACKEND, 0, 0, 0));
    sources.push_back(M::StreamSource<TYPE>(SOURCE1, BACKEND, 1, 0, 1));
    sources.push_back(M::StreamSource<TYPE>(SOURCE2, BACKEND, 2, 1, 0));
    sources.push_back(M::StreamSource<TYPE>(SOURCE3, BACKEND, 3, 1, 1));

    M::VideoStream<TYPE> stream(sources);
    stream.set_param(VISUAL_WIDTH, VISUAL_HEIGHT, 2, 2);
    cv::Mat merged_frame(cv::Size(VISUAL_WIDTH * 2, VISUAL_HEIGHT * 2), CV_8UC3);
    cv::Mat display_frame;

    if (!stream.isOpened())
    {
#ifdef DEBUG
        std::cout << "[DEBUG] "
                  << "[API][ERROR] Error opening video stream or file" << std::endl;
#endif // DEBUG
    }

    while (1)
    {
        std::string command;
#ifdef JSON
        nlohmann::json j;
#endif // JSON
        std::cout << "Enter COMMAND:" << std::endl;
        std::cout.flush();
        std::cin >> command;
        if (command == std::string("quit"))
        {
            return 0;
        }
        stream.grab_frame();
        cv::Mat merged(cv::Size(VISUAL_WIDTH * 2, VISUAL_HEIGHT * 2), CV_8UC3);
        cv::Mat display_frame;
        M::pipeline_data p_data(merged);
        bool state = stream.retrieve_frame(p_data.cap_frame);
        // if (!state)
        //     break;
        p_data.uuid = gen_uuid(save_dir, std::string(".jpg"));
        for (int i = 0; i < 4; i++)
        {
            cv::Mat roi(p_data.cap_frame, cv::Rect((i % 2) * VISUAL_WIDTH, (i / 2) * VISUAL_HEIGHT, VISUAL_WIDTH, VISUAL_HEIGHT));
#ifdef INFERENCE_DARKNET
            std::vector<bbox_t> result = yolo.detect_cv(roi, thresh, true);
#else
            std::vector<YOLOv4::DetectRes> result = yolo->EngineInference(roi);
#endif // INFERENCE_DARKNET
#ifdef DEBUG
            std::cout << "[DEBUG] " << p_data.uuid << "\tresult.size\t" << result.size() << "\tcam: " << i << std::endl;
#endif // DEBUG
#ifdef INFERENCE_DARKNET
            std::pair<int, std::vector<bbox_t>> _pair(i, result);
#else
            std::pair<int, std::vector<YOLOv4::DetectRes>> _pair(i, result);
#endif // INFERENCE_DARKNET
            p_data.result.insert(_pair);
        }

        //Write image to database
        bool write_result = cv::imwrite(p_data.uuid, p_data.cap_frame);
#ifdef DEBUG
        std::cout << "[DEBUG] " << p_data.uuid << "\t" << p_data.cap_frame.size << std::endl;
#endif // DEBUG
        std::cout << "[FILENAME] " << (write_result ? p_data.uuid : "") << std::endl;
#ifdef JSON
        j["filename"] = write_result ? p_data.uuid : "";
#endif // JSON
#ifdef INFERENCE_DARKNET
        for (const std::pair<const int, std::vector<bbox_t>> &_pair : p_data.result)
        {
            int cam = _pair.first;
            std::vector<bbox_t> result = _pair.second;
            cv::Mat roi(p_data.cap_frame, cv::Rect((cam % 2) * VISUAL_WIDTH, (cam / 2) * VISUAL_HEIGHT, VISUAL_WIDTH, VISUAL_HEIGHT));
#ifdef INFERENCE_ALPHAPOSE_TORCH
            std::vector<pose_box> inputPose;
#endif // INFERENCE_ALPHAPOSE_TORCH
#ifdef DEBUG
            std::cout << "[DEBUG] "
                      << "UUID\t" << p_data.uuid << std::endl;
#endif // DEBUG
            for (const bbox_t &rect : result)
            {
                int x_left = (rect.x < 0) ? 0 : rect.x;
                int y_top = (rect.y < 0) ? 0 : rect.y;
#ifndef JSON
                std::cout << "[CAM] " << cam << "\t"
                          << obj_names[rect.obj_id]
                          << "\t" << static_cast<int>(rect.prob * 100) << "%\t"
                          << "x_left:  " << static_cast<int>(x_left)
                          << "   y_top:  " << static_cast<int>(y_top)
                          << "   width:  " << static_cast<int>(rect.w)
                          << "   height:  " << static_cast<int>(rect.h)
                          << std::endl;
                std::cout.flush();
#endif // !JSON
#ifdef DEBUG
                std::ostringstream os;
                os << "cam:" << cam << "  classes:" << obj_names[rect.obj_id] << "  x:" << rect.x << "  y:" << rect.y << "  w:" << rect.w << "  h:" << rect.h << "  prob:" << rect.prob;
                std::cout << "[DEBUG] " << os.str() << std::endl;
#endif // DEBUG

                // if (!dont_show)
                {
                    char t[256];
                    sprintf(t, "%.2f", rect.prob);
                    std::string name = obj_names[rect.obj_id] + "-" + t;
                    cv::putText(roi, name, cv::Point(x_left, y_top - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, obj_id_to_color(rect.obj_id), 2);
                    cv::Rect rst(x_left, y_top, rect.w, rect.h);
                    cv::rectangle(roi, rst, obj_id_to_color(rect.obj_id), 2, cv::LINE_8, 0);
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
#ifdef INFERENCE_ALPHAPOSE_TORCH
            std::vector<PoseKeypoints> pKps;
            al->predict(roi, inputPose, pKps);
            al->draw(roi, pKps);
#endif // INFERENCE_ALPHAPOSE_TORCH
#ifdef JSON
            std::ostringstream os;
            os << "CAM" << cam;
            j[os.str()] = M::res_to_json(result
#ifdef INFERENCE_ALPHAPOSE_TORCH
                                         ,
                                         pKps
#endif // INFERENCE_ALPHAPOSE_TORCH
            );
#endif // JSON
        }
#else

        for (const std::pair<const int, std::vector<YOLOv4::DetectRes>> &_pair : p_data.result)
        {
            int cam = _pair.first;
            std::vector<YOLOv4::DetectRes> result = _pair.second;
            cv::Mat roi(p_data.cap_frame, cv::Rect((cam % 2) * VISUAL_WIDTH, (cam / 2) * VISUAL_HEIGHT, VISUAL_WIDTH, VISUAL_HEIGHT));
#ifdef INFERENCE_ALPHAPOSE_TORCH
            std::vector<pose_box> inputPose;
#endif // INFERENCE_ALPHAPOSE_TORCH
#ifdef DEBUG
            std::cout << "[DEBUG] "
                      << "UUID\t" << p_data.uuid << std::endl;
#endif // DEBUG
            for (const auto &rect : result)
            {
                int x_left = rect.x - rect.w / 2;
                x_left = (x_left < 0) ? 0 : x_left;
                int y_top = rect.y - rect.h / 2;
                y_top = (y_top < 0) ? 0 : y_top;
#ifndef JSON
                std::cout << "[CAM] " << cam << "\t"
                          << yolo->detect_labels[rect.classes]
                          << "\t" << static_cast<int>(rect.prob * 100) << "%\t"
                          << "x_left:  " << static_cast<int>(x_left)
                          << "   y_top:  " << static_cast<int>(y_top)
                          << "   width:  " << static_cast<int>(rect.w)
                          << "   height:  " << static_cast<int>(rect.h)
                          << std::endl;
                std::cout.flush();
#endif // JSON
#ifdef DEBUG
                std::ostringstream os;
                os << "cam:" << cam << "  classes:" << rect.classes << "  x:" << rect.x << "  y:" << rect.y << "  w:" << rect.w << "  h:" << rect.h << "  prob:" << rect.prob;
                std::cout << "[DEBUG] " << os.str() << std::endl;
#endif // DEBUG

                // if (!dont_show)
                {
                    char t[256];
                    sprintf(t, "%.2f", rect.prob);
                    std::string name = yolo->detect_labels[rect.classes] + "-" + t;
                    cv::putText(roi, name, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, yolo->class_colors[rect.classes], 2);
                    cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
                    cv::rectangle(roi, rst, yolo->class_colors[rect.classes], 2, cv::LINE_8, 0);
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
#ifdef INFERENCE_ALPHAPOSE_TORCH
            std::vector<PoseKeypoints> pKps;
            al->predict(roi, inputPose, pKps);
            al->draw(roi, pKps);
#ifdef INFERENCE_TABULAR_TORCH
            std::vector<int> tabular_pred;
            tab->predict(inputPose, pKps, tabular_pred);
#endif // INFERENCE_TABULAR_TORCH
#endif // INFERENCE_ALPHAPOSE_TORCH
#ifdef JSON
            std::ostringstream os;
            os << "CAM" << cam;
            j[os.str()] = M::res_to_json(result
#ifdef INFERENCE_ALPHAPOSE_TORCH
                                         ,
                                         pKps
#ifdef INFERENCE_TABULAR_TORCH
                                         ,
                                         tabular_pred
#endif // INFERENCE_TABULAR_TORCH
#endif // INFERENCE_ALPHAPOSE_TORCH
            );
#endif // JSON
        }
#endif // INFERENCE_DARKNET
#ifdef JSON
        std::string json_file = write_result ? p_data.uuid : "";
        mreplace(json_file, ".jpg", ".json");
        std::ofstream file(json_file);
        file << j.dump(3);
        file.close();
        // std::cout << "[JSON] " << j << std::endl;
        std::cout.flush();
#endif // JSON
        cv::resize(p_data.cap_frame, display_frame, cv::Size(1600, 900));
        if (!dont_show)
        {
            // If the frame is empty, break immediately
            cv::imshow("final", display_frame);
            // Press  ESC on keyboard to exit
            char c = (char)cv::waitKey(0);
        }
        else
            cv::imwrite("result.jpg", display_frame);

        merged.release();
        display_frame.release();
    }

    return 0;
}
#endif // NOBI_CAMERA_AI_API
