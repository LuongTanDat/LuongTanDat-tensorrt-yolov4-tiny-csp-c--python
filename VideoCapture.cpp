#include "VideoCapture.h"

// #define SINGLE_CAM
// #define FOUR_CAMS

#ifdef SINGLE_CAM
int main()
{
    M::VideoCapture cap(0, 0, 0);
#ifdef VIDEO_EXAMPLES
    cap.open("../EMoi.mp4", cv::CAP_GSTREAMER);
#endif // VIDEO_EXAMPLES
#ifdef CAM_ID_EXAMPLES
    cap.open(0, cv::CAP_V4L2);
#endif // CAM_ID_EXAMPLES
    while (true)
    {
        cv::Mat frame;
        cv::Mat black = cv::Mat(cv::Size(VISUAL_WIDTH, VISUAL_HEIGHT), CV_8UC3, cv::Scalar(0, 0, 0));
        if (!cap.read(frame))
        {
#ifdef DEBUG
            std::cout << "Cam " << cap.getCamId() << "\t"
                      << "frame.empty()" << std::endl;
#endif // DEBUG
        }
        else
        {
            cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
            cv::resize(frame, black, cv::Size(VISUAL_WIDTH, VISUAL_HEIGHT));
        }

        cv::imshow("HNIW", black);
        // Press  ESC on keyboard to exit
        char c = (char)cv::waitKey(25);
        if (c == 27)
        {
            break;
        }
    }
}
#endif // SINGLE_CAM

#ifdef FOUR_CAMS
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
bool stop_video = false;
M::send_one_replaceable_object<M::pipeline_data> stream2show;

int main()
{
#ifdef __linux__
    // XInitThreads();
#elif _WIN32
#endif
    std::vector<M::StreamSource<TYPE>> sources;
    sources.push_back(M::StreamSource<TYPE>(SOURCE2, BACKEND, 2, 1, 0));
    sources.push_back(M::StreamSource<TYPE>(SOURCE0, BACKEND, 0, 0, 0));
    sources.push_back(M::StreamSource<TYPE>(SOURCE1, BACKEND, 1, 0, 1));
    sources.push_back(M::StreamSource<TYPE>(SOURCE3, BACKEND, 3, 1, 1));

    M::VideoStream<TYPE> stream(sources);
    stream.set_param(VISUAL_WIDTH, VISUAL_HEIGHT, 2, 2);
    M::pipeline_data frame_merged;

    if (!stream.isOpened())
    {
#ifdef DEBUG
        std::cout << "[DEBUG] " << "Error opening video stream or file" << std::endl;
#endif // DEBUG
        stop_video = true;
    }
    else
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
                stream2show.send(p_data);
            }
        });

    while (stop_video)
    {
        // stream.grab_frame();
        // bool state = stream.retrieve_frame(merged);
        frame_merged = stream2show.receive();
        cv::Mat EMoi;
        cv::resize(frame_merged.cap_frame, EMoi, cv::Size(960, 720));
        cv::imshow("EMoi", EMoi);
        char c = (char)cv::waitKey(1);
        if (c == 27)
            break;
    }

    stop_video = false;
    cv::destroyAllWindows();
    stream.release();

    if (retrieve_frame_thead.joinable())
        retrieve_frame_thead.join();
    return 0;
}
#endif // FOUR_CAMS
