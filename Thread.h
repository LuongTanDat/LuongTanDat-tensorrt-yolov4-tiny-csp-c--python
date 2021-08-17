#ifndef THREAD_H
#define THREAD_H

// #define INFERENCE_DARKNET
// #define INFERENCE_ALPHAPOSE_TORCH
// #define JSON

// #define INFERENCE_VIDEO
// #define TENSORRT_API
// #define NOBI_CAMERA_AI_API
// #define DEBUG

// #define CAM_ID_EXAMPLES
// #define VIDEO_EXAMPLES

#define YOLOv4_CSP_512
// #define YOLOv4_608
// #define YOLOv4_512_VIZGARD
// #define YOLOv4_768_VIZGARD

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
#endif // INFERENCE_ALPHAPOSE_TORCH

namespace M
{
    nlohmann::json bbox_to_json(YOLOv4::DetectRes &res)
    {
        nlohmann::json j;
        j["bbox"] = nlohmann::json::array({res.x - res.w / 2, res.y - res.h / 2, res.w, res.h});
        j["cls"] = res.classes;
        j["prob"] = res.prob;
        j["feat"] = {};
        for (int i = 0; i < 7; i++)
        {
            j["feat"].push_back(*(res.feature + i));
        }
        return j;
    }

    nlohmann::json res_to_json(std::vector<YOLOv4::DetectRes> &vecBBox, std::vector<PoseKeypoints> &vecKp)
    {
        nlohmann::json j = {};
        int countKp = 0;
        for (int i = 0; i < vecBBox.size(); i++)
        {
            nlohmann::json jx;
            jx["det"] = bbox_to_json(vecBBox[i]);
            if (vecBBox[i].classes > 4)
            {
                jx["pose"] = nlohmann::json::value_t::null;
            }
            else
            {
                jx["pose"] = vecKp[countKp].to_json();
                countKp++;
            }
            j.push_back(jx);
        }
        return j;
    }

    void convert_DetectRes_bbox(const YOLOv4::DetectRes &res, bbox &out)
    {
        out.rect.x = (float)(res.x - res.w / 2);
        out.rect.y = (float)(res.y - res.h / 2);
        out.rect.width = (float)res.w;
        out.rect.height = (float)res.h;
        out.score = (float)res.prob;
    }

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

    // TODO: Minimize, optimize this
    struct pipeline_data
    {
        cv::Mat cap_frame;
        // cv::Mat frame_from_cam[4];
        std::map<int, std::vector<YOLOv4::DetectRes>> result;
        std::map<int, std::vector<PoseKeypoints>> poseKeypoints;
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
