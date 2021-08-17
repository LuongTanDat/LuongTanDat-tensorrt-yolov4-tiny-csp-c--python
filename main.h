#ifndef MAIN_H
#define MAIN_H

#include "Thread.h"
#include "VideoCapture.h"
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <iostream>
#include <fstream>

void mreplace(std::string &input, std::string sub_string, std::string new_string, int count = -1);
void mreplace(std::string &input, std::string sub_string, std::string new_string, int count)
{
    if (count >= 0)
    {
        for (int i = 0; i < count; i++)
        {
            size_t pos = input.find(sub_string);
            if (pos == std::string::npos)
                break;
            input.replace(pos, sub_string.length(), new_string);
        }
    }
    else
    {
        while (true)
        {
            size_t pos = input.find(sub_string);
            if (pos == std::string::npos)
                break;
            input.replace(pos, sub_string.length(), new_string);
        }
    }
}

std::string gen_uuid(std::string prefix = std::string("./"), std::string postfix = std::string(""));
std::string gen_uuid(std::string prefix, std::string postfix)
{
    boost::uuids::random_generator gen;
    std::ostringstream os;

    os << prefix << "/" << gen() << postfix;
    std::string id = os.str();
    // mreplace(id, std::string("-"), std::string(""));
    mreplace(id, std::string("//"), std::string("/"));
    return id;
}

void ShowHelpAndExit(const char *szBadOption = NULL)
{
    bool bThrowError = false;
    std::ostringstream oss;
    if (szBadOption)
    {
        bThrowError = true;
        oss << "Error parsing \"" << szBadOption << "\"" << std::endl;
    }
    oss << "Options:" << std::endl
        << "    --engine-file   : TRT model" << std::endl
        << "    --label-file    : .names file" << std::endl
        << "    --dims          : W H format" << std::endl
        << "    --type-yolo     : yolov4/csp/tiny" << std::endl
#ifdef INFERENCE_ALPHAPOSE_TORCH
        << "    --alphapose-jit : Alphapose torchscript model" << std::endl
#endif // INFERENCE_ALPHAPOSE_TORCH
#ifdef NOBI_CAMERA_AI_API
        << "    --save-dir      : Path to folder contain images" << std::endl
#endif
        << "    --obj-thres     : object threshold" << std::endl
        << "    --nms-thres     : non maximize suppressor threshold" << std::endl
        << "    --dont-show     : del show image by opencv" << std::endl;
    oss << std::endl;
    if (bThrowError)
    {
        throw std::invalid_argument(oss.str());
    }
}

void ParseCommandLine(int argc, char *argv[], Config *config, bool &dont_show, std::string &save_dir, std::string &alphapose_model)
{
    int i;

    for (i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == std::string("--help"))
        {
            ShowHelpAndExit();
        }
        else if (std::string(argv[i]) == std::string("--engine-file"))
        {
            if (++i == argc)
                ShowHelpAndExit("--engine-file");
            else
                config->engine_file = std::string(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--save-dir"))
        {
            if (++i == argc)
                ShowHelpAndExit("--save-dir");
            else
                save_dir = std::string(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--alphapose-jit"))
        {
            if (++i == argc)
                ShowHelpAndExit("--alphapose-jit");
            else
                alphapose_model = std::string(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--label-file"))
        {
            if (++i == argc)
                ShowHelpAndExit("--label-file");
            else
                config->labels_file = std::string(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--dims"))
        {
            if (++i == argc /* || 2 != sscanf(argv[i], "%dx%d", &config->IMAGE_HEIGHT, &config->IMAGE_WIDTH)*/)
                ShowHelpAndExit("--dims");
            else
                config->IMAGE_WIDTH = std::stoi(std::string(argv[i]));

            if (++i == argc /* || 2 != sscanf(argv[i], "%dx%d", &config->IMAGE_HEIGHT, &config->IMAGE_WIDTH)*/)
                ShowHelpAndExit("--dims");
            else
                config->IMAGE_HEIGHT = std::stoi(std::string(argv[i]));
            continue;
        }
        else if (std::string(argv[i]) == std::string("--obj-thres"))
        {
            if (++i == argc)
                ShowHelpAndExit("--obj-thres");
            else
                config->obj_threshold = std::stof(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--nms-thres"))
        {
            if (++i == argc)
                ShowHelpAndExit("--nms-thres");
            else
                config->nms_threshold = std::stof(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--type-yolo"))
        {
            if (++i == argc)
                ShowHelpAndExit("--type-yolo");
            else
                config->model = std::string(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--dont-show"))
        {
            dont_show = true;
            continue;
        }
        else if (std::string(argv[i]) == std::string("--HNIW"))
        {
            if (++i == argc /* || 2 != sscanf(argv[i], "%dx%d", &config->IMAGE_HEIGHT, &config->IMAGE_WIDTH)*/)
            {
                ShowHelpAndExit("--dims");
            }
            else
                config->IMAGE_WIDTH = std::stoi(std::string(argv[i]));

            if (++i == argc /* || 2 != sscanf(argv[i], "%dx%d", &config->IMAGE_HEIGHT, &config->IMAGE_WIDTH)*/)
            {
                ShowHelpAndExit("--dims");
            }
            else
                config->IMAGE_HEIGHT = std::stoi(std::string(argv[i]));
            continue;
        }
        else
        {
            ShowHelpAndExit((std::string("input not include ") + std::string(argv[i])).c_str());
        }
    }
}

// /mnt/1882C07482C05840/TensorRT/scaled-c++/build/Yolov4_trt --engine-file "/mnt/1882C07482C05840/TensorRT/model-zoo/yolov4-csp/yolov4-csp-512.engine" \
--label-file "/mnt/1882C07482C05840/TensorRT/model-zoo/yolov4-csp/yolov4-csp-512.names" \
--dims 512 512 --obj-thres 0.5 --nms-thres 0.45 --type-yolo yolov4 --dont-show
#endif // MAIN_H
