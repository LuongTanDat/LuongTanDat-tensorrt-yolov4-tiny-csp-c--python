#ifndef MAIN_H
#define MAIN_H

#include "Thread.h"
#include "VideoCapture.h"
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <iostream>
#include <fstream>
#include "crow_all.h"
#include "base64.h"

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

#ifdef INFERENCE_DARKNET
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
        << "    --weights-file  : Weights model" << std::endl
        << "    --cfg-file      : .cfg file" << std::endl
        << "    --names-file    : .names file" << std::endl
#ifdef INFERENCE_ALPHAPOSE_TORCH
        << "    --alphapose-jit : Alphapose torchscript model" << std::endl
#endif // INFERENCE_ALPHAPOSE_TORCH
        << "    --port          : PORT" << std::endl
        << "    --thresh        : object threshold" << std::endl
        << "    --dont-show     : del show image by opencv" << std::endl;
    oss << std::endl;
    if (bThrowError)
    {
        throw std::invalid_argument(oss.str());
    }
}

void ParseCommandLine(int argc, char *argv[], std::string &weights_file, std::string &names_file, std::string &cfg_file, unsigned int &port
#ifdef INFERENCE_ALPHAPOSE_TORCH
                      ,
                      std::string &alphapose_model
#endif // INFERENCE_ALPHAPOSE_TORCH
                      ,
                      float &thresh, bool &dont_show)
{
    int i;
    for (i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == std::string("--help"))
        {
            ShowHelpAndExit();
        }
        else if (std::string(argv[i]) == std::string("--weights-file"))
        {
            if (++i == argc)
                ShowHelpAndExit("--weights-file");
            else
                weights_file = std::string(argv[i]);
            continue;
        }

        else if (std::string(argv[i]) == std::string("--port"))
        {
            if (++i == argc)
                ShowHelpAndExit("--port");
            else
                port = std::stoi(argv[i]);
            continue;
        }
#ifdef INFERENCE_ALPHAPOSE_TORCH
        else if (std::string(argv[i]) == std::string("--alphapose-jit"))
        {
            if (++i == argc)
                ShowHelpAndExit("--alphapose-jit");
            else
                alphapose_model = std::string(argv[i]);
            continue;
        }
#endif // INFERENCE_ALPHAPOSE_TORCH
        else if (std::string(argv[i]) == std::string("--names-file"))
        {
            if (++i == argc)
                ShowHelpAndExit("--names-file");
            else
                names_file = std::string(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--cfg-file"))
        {
            if (++i == argc)
                ShowHelpAndExit("--cfg-file");
            else
                cfg_file = std::string(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--thresh"))
        {
            if (++i == argc)
                ShowHelpAndExit("--thresh");
            else
                thresh = std::stof(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--dont-show"))
        {
            dont_show = true;
            continue;
        }
        else
        {
            ShowHelpAndExit((std::string("input not include ") + std::string(argv[i])).c_str());
        }
    }
}

std::vector<std::string> objects_names_from_file(std::string const filename)
{
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open())
        return file_lines;
    for (std::string line; getline(file, line);)
        file_lines.push_back(line);
    return file_lines;
}
#else
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
#ifdef INFERENCE_TABULAR_TORCH
        << "    --tabular-jit   : Tabular learner torchscript model" << std::endl
#endif // INFERENCE_TABULAR_TORCH
#endif // INFERENCE_ALPHAPOSE_TORCH
        << "    --port          : PORT" << std::endl
        << "    --obj-thres     : object threshold" << std::endl
        << "    --nms-thres     : non maximize suppressor threshold" << std::endl
        << "    --dont-show     : del show image by opencv" << std::endl;
    oss << std::endl;
    if (bThrowError)
    {
        throw std::invalid_argument(oss.str());
    }
}

void ParseCommandLine(int argc, char *argv[], Config *config, bool &dont_show, unsigned int &port
#ifdef INFERENCE_ALPHAPOSE_TORCH
                      ,
                      std::string &alphapose_model
#ifdef INFERENCE_TABULAR_TORCH
                      ,
                      std::string &tabular_model
#endif // INFERENCE_TABULAR_TORCH
#endif // INFERENCE_ALPHAPOSE_TORCH
)
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
        else if (std::string(argv[i]) == std::string("--port"))
        {
            if (++i == argc)
                ShowHelpAndExit("--port");
            else
                port = std::stoi(argv[i]);
            continue;
        }
#ifdef INFERENCE_ALPHAPOSE_TORCH
        else if (std::string(argv[i]) == std::string("--alphapose-jit"))
        {
            if (++i == argc)
                ShowHelpAndExit("--alphapose-jit");
            else
                alphapose_model = std::string(argv[i]);
            continue;
        }
#ifdef INFERENCE_TABULAR_TORCH
        else if (std::string(argv[i]) == std::string("--tabular-jit"))
        {
            if (++i == argc)
                ShowHelpAndExit("--tabular-jit");
            else
                tabular_model = std::string(argv[i]);
            continue;
        }
#endif // INFERENCE_TABULAR_TORCH
#endif // INFERENCE_ALPHAPOSE_TORCH
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
#endif // INFERENCE_DARKNET

// /mnt/1882C07482C05840/TensorRT/scaled-c++/build/Yolov4_trt --engine-file "/mnt/1882C07482C05840/TensorRT/model-zoo/yolov4-csp/yolov4-csp-512.engine" \
--label-file "/mnt/1882C07482C05840/TensorRT/model-zoo/yolov4-csp/yolov4-csp-512.names" \
--dims 512 512 --obj-thres 0.5 --nms-thres 0.45 --type-yolo yolov4 --dont-show
#endif // MAIN_H
