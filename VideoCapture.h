#ifndef VIDEO_CAPTURE_H
#define VIDEO_CAPTURE_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>
#include <assert.h>
#include "Thread.h"

#define VISUAL_WIDTH 1280
#define VISUAL_HEIGHT 720
#define MAX_VISUAL_CAMERAS 4
#define NUMBER_OF_VISUAL_CAMERAS MAX_VISUAL_CAMERAS

namespace M
{
    template <typename T>
    struct StreamSource
    {
        StreamSource(T src, int bk, int cam, int row, int col) : source(src), backend(bk), cam_id(cam), row_id(row), col_id(col) {}

        T source;
        int backend;
        int cam_id;
        int row_id;
        int col_id;
    };

    class VideoCapture : public cv::VideoCapture
    {
    public:
        VideoCapture(int index, int cam_id, int row_id, int col_id, int apiPreference) : cv::VideoCapture(index, apiPreference)
        {
            this->cam_id = cam_id;
            this->row_id = row_id;
            this->col_id = col_id;
        }

        VideoCapture(const std::string &filename, int cam_id, int row_id, int col_id, int apiPreference) : cv::VideoCapture(filename, apiPreference)
        {
            this->cam_id = cam_id;
            this->row_id = row_id;
            this->col_id = col_id;
        }

        VideoCapture(int cam_id, int row_id, int col_id)
        {
            this->cam_id = cam_id;
            this->row_id = row_id;
            this->col_id = col_id;
        }

        int getCamId()
        {
            return cam_id;
        }

        int getRowId()
        {
            return row_id;
        }

        int getColId()
        {
            return col_id;
        }

    private:
        int cam_id;
        int row_id;
        int col_id;
    };

    template <typename T>
    class VideoStream
    {
    public:
        VideoStream(std::vector<StreamSource<T>> sources)
        {
            this->setupCapture(sources);
        }

        VideoStream(std::vector<StreamSource<T>> sources, int visual_width, int visual_height, int num_columns_camera, int num_rows_camera)
        {
            this->setupCapture(sources);
            assert(this->set_param(visual_width, visual_height, num_columns_camera, num_rows_camera));
        }

        ~VideoStream()
        {
            for (VideoCapture cap : this->captures)
            {
                cap.release();
            }
        }

        bool set_param(int visual_width, int visual_height, int num_columns_camera, int num_rows_camera)
        {
            this->visual_width = visual_width;
            this->visual_height = visual_height;
            this->num_columns_camera = num_columns_camera;
            this->num_rows_camera = num_rows_camera;

            for (VideoCapture cap : this->captures)
            {
                if ((cap.getColId() >= this->num_columns_camera) || (cap.getRowId() >= this->num_rows_camera))
                {
                    return false;
                }
            }
            return true;
        }

        void grab_frame()
        {
            for (VideoCapture cap : this->captures)
            {
                cap.grab();
            }
        }

        bool retrieve_frame(cv::Mat &returned)
        {
            bool state = true;
            cv::Mat merged(cv::Size(this->visual_width * this->num_columns_camera, this->visual_height * this->num_rows_camera), CV_8UC3);
            // cv::Mat multi_frame[4];
            // Capture frame-by-frame
            for (VideoCapture cap : this->captures)
            {
                cv::Mat frame; //, resized_frame;
                cv::Mat roi(merged, cv::Rect(cap.getColId() * this->visual_width, cap.getRowId() * this->visual_height, this->visual_width, this->visual_height));

                if (!cap.read(frame))
                {
#ifdef DEBUG
                    std::cout << "Cam " << cap.getCamId() << "\t"
                              << "frame.empty()" << std::endl;
#endif // DEBUG
                    cv::Mat black = cv::Mat(cv::Size(this->visual_width, this->visual_height), CV_8UC3, cv::Scalar(0, 0, 0));
                    black.copyTo(roi);
                    state &= false;
                }
                else
                {
                    cv::resize(frame, roi, cv::Size(this->visual_width, this->visual_height));
                    state &= true;
                }
            }
            merged.copyTo(returned);
            return state;
        }

        void start()
        {
        }

        bool isOpened()
        {
            bool state = true;
            for (VideoCapture cap : this->captures)
            {
                state &= cap.isOpened();
            }
            return state;
        }

        void release()
        {
            for (VideoCapture cap : this->captures)
            {
                cap.release();
            }
        }

    private:
        void setupCapture(std::vector<StreamSource<T>> sources)
        {
            for (auto src : sources)
            {
                VideoCapture cap(src.cam_id, src.row_id, src.col_id);
                cap.open(src.source, src.backend);
                this->captures.push_back(cap);
            }
        }
        std::vector<VideoCapture> captures;
        int visual_width;
        int visual_height;
        int num_columns_camera;
        int num_rows_camera;
    };
}

#endif // VIDEO_CAPTURE_H
