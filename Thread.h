#ifndef THREAD_H
#define THREAD_H

// #define INFERENCE_DARKNET
// #define INFERENCE_ALPHAPOSE_TORCH
// #define INFERENCE_TABULAR_TORCH
// #define JSON

// #define INFERENCE_VIDEO
// #define TENSORRT_API
// #define NOBI_CAMERA_AI_API
// #define DEBUG

// #define CAM_ID_EXAMPLES
// #define VIDEO_EXAMPLES

#define YOLOv4_CSP_512
// #define YOLOv4_608

#include <thread>
#include <chrono>
#include <atomic>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <cctype>
#ifdef INFERENCE_DARKNET
#include "yolo_v2_class.hpp"
#else
#include "Yolov4.h"
#endif // INFERENCE_DARKNET
#ifdef JSON
#include "json.hpp"
#endif // JSON
#ifdef INFERENCE_ALPHAPOSE_TORCH
#include "AlphaPose.h"
#ifdef INFERENCE_TABULAR_TORCH
#include "Tabular.h"
#endif // INFERENCE_TABULAR_TORCH
#endif // INFERENCE_ALPHAPOSE_TORCH

namespace M
{
#ifdef JSON
#ifdef INFERENCE_DARKNET
    nlohmann::json bbox_to_json(bbox_t &res)
    {
        nlohmann::json j;
        j["bbox"] = nlohmann::json::array({res.x, res.y, res.w, res.h});
        j["cls"] = res.obj_id;
        j["prob"] = res.prob;
#ifdef DEBUG
        std::cout << "[DEBUG][JSON] " << j << std::endl;
#endif // DEBUG
        return j;
    }
#else
    nlohmann::json bbox_to_json(YOLOv4::DetectRes &res)
    {
        nlohmann::json j;
        j["bbox"] = nlohmann::json::array({res.x - res.w / 2, res.y - res.h / 2, res.w, res.h});
        j["cls"] = res.classes;
        j["prob"] = res.prob;
#ifdef DEBUG
        std::cout << "[DEBUG][JSON] " << j << std::endl;
#endif // DEBUG
        return j;
    }
#endif // INFERENCE_DARKNET

#ifdef INFERENCE_DARKNET
    nlohmann::json res_to_json(std::vector<bbox_t> &vecBBox
#else
    nlohmann::json res_to_json(std::vector<YOLOv4::DetectRes> &vecBBox
#endif // INFERENCE_DARKNET
#ifdef INFERENCE_ALPHAPOSE_TORCH
                               ,
                               std::vector<PoseKeypoints> &vecKp
#ifdef INFERENCE_TABULAR_TORCH
                               ,
                               std::vector<int32_t> tabular_pred
#endif // INFERENCE_TABULAR_TORCH
#endif // INFERENCE_ALPHAPOSE_TORCH
    )
    {
        nlohmann::json j = {};
#ifdef INFERENCE_ALPHAPOSE_TORCH
        int countKp = 0;
#endif // INFERENCE_ALPHAPOSE_TORCH
        for (int i = 0; i < vecBBox.size(); i++)
        {
            nlohmann::json jx;
            jx["det"] = bbox_to_json(vecBBox[i]);
#ifdef INFERENCE_ALPHAPOSE_TORCH
#ifdef INFERENCE_DARKNET
            if (vecBBox[i].obj_id > 4)
#else
            if (vecBBox[i].classes > 4)
#endif // INFERENCE_DARKNET
            {
                jx["pose"] = nlohmann::json::value_t::null;
            }
            else
            {
                jx["pose"] = vecKp[countKp].to_json();
#ifdef INFERENCE_TABULAR_TORCH
                jx["tab"] = tabular_pred[countKp];
#endif // INFERENCE_TABULAR_TORCH
                countKp++;
            }
#endif // INFERENCE_ALPHAPOSE_TORCH
            j.push_back(jx);
        }
        return j;
    }
#endif // JSON

#ifdef INFERENCE_ALPHAPOSE_TORCH
#ifdef INFERENCE_DARKNET
    void convert_DetectRes_bbox(const bbox_t &res, bbox &out)
#else
    void convert_DetectRes_bbox(const YOLOv4::DetectRes &res, bbox &out)
#endif // INFERENCE_DARKNET
    {
#ifdef INFERENCE_DARKNET
        out.rect.x = (float)res.x;
        out.rect.y = (float)res.y;
#else
        out.rect.x = (float)(res.x - res.w / 2);
        out.rect.y = (float)(res.y - res.h / 2);
#ifdef INFERENCE_TABULAR_TORCH
        memcpy(out.feat, res.feature, 7 * sizeof(float));
#endif // INFERENCE_TABULAR_TORCH
#endif // INFERENCE_DARKNET
        out.rect.width = (float)res.w;
        out.rect.height = (float)res.h;
        out.score = (float)res.prob;
    }

#ifdef INFERENCE_DARKNET
    void convert_vecDetectRes_vecbbox(const std::vector<bbox_t> &vecRes, std::vector<bbox> &conv)
    {
        for (bbox_t res : vecRes)
        {
            if (res.obj_id > 4)
                continue;

            bbox outres;
            convert_DetectRes_bbox(res, outres);
            conv.push_back(outres);
        }
    }
#else
    void convert_vecDetectRes_vecbbox(const std::vector<YOLOv4::DetectRes> &vecRes, std::vector<bbox> &conv)
    {
        for (YOLOv4::DetectRes res : vecRes)
        {
            if (res.classes > 4)
                continue;

            bbox outres;
            convert_DetectRes_bbox(res, outres);
            conv.push_back(outres);
        }
    }
#endif // INFERENCE_DARKNET
#endif // INFERENCE_ALPHAPOSE_TORCH

    // TODO: Minimize, optimize this
    struct pipeline_data
    {
        cv::Mat cap_frame;
        // cv::Mat frame_from_cam[4];
#ifdef INFERENCE_DARKNET
        std::map<int, std::vector<bbox_t>> result;
#else
        std::map<int, std::vector<YOLOv4::DetectRes>> result;
#endif // INFERENCE_DARKNET
#ifdef INFERENCE_ALPHAPOSE_TORCH
        std::map<int, std::vector<PoseKeypoints>> poseKeypoints;
#endif // INFERENCE_ALPHAPOSE_TORCH
        std::string uuid;
        pipeline_data(){};
        pipeline_data(cv::Mat frame) : cap_frame(frame){};
    };

    template <typename T>
    class send_one_replaceable_object
    {
        std::atomic<T *> a_ptr = {nullptr};

    public:
        void send(T const &_obj)
        {
            T *new_ptr = new T;
            *new_ptr = _obj;
            // TODO: The `unique_ptr` prevents a scary memory leak, why?
            std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
        }

        T receive()
        {
            std::unique_ptr<T> ptr;
            do
            {
                while (!a_ptr)
                    std::this_thread::sleep_for(std::chrono::milliseconds(3));
                ptr.reset(a_ptr.exchange(nullptr));
            } while (!ptr);
            return *ptr;
        }
    };

    bool isdigit(std::string str)
    {
        bool state = true;
        for (int i = 0; i < str.length(); i++)
        {
            if (!std::isdigit(str[i]))
            {
                state &= false;
            }
        }
        return state;
    }
}

#endif // THREAD_H
