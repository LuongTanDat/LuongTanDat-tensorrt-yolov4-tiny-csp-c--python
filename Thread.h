#ifndef THREAD_H
#define THREAD_H

// #define INFERENCE_DARKNET
#define INFERENCE_ALPHAPOSE_TORCH

#define INFERENCE_VIDEO
// #define TENSORRT_API
// #define NOBI_CAMERA_AI_API
// #define DEBUG

// #define CAM_ID_EXAMPLES
#define VIDEO_EXAMPLES

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

#ifdef INFERENCE_ALPHAPOSE_TORCH
#include "AlphaPose.h"
#endif // INFERENCE_ALPHAPOSE_TORCH

#ifdef __linux__
#include <X11/Xlib.h>
#elif _WIN32
#endif

namespace M
{
#ifdef INFERENCE_ALPHAPOSE_TORCH
    /*
        T: bbox_t or YOLOv4::DetectRes
    */
    template <typename T>
    void convert_DetectRes_bbox(const T &res, bbox &out)
    {
        return bbox((float)res.x, (float)res.y, (float)res.w, (float)res.h, (float)res.prob);
    }

    template <typename T>
    void convert_vecDetectRes_vecbbox(const std::vector<T> &vecRes, std::vector<bbox> &conv)
    {
        for (T res : vecRes)
        {
            bbox outres;
            convert_DetectRes_bbox<T>(res, outres);
            conv.push_back(outres);
        }
    }
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
