#pragma once
#include "yolov3.hpp"
#include "yolov8.hpp"
#include "img.hpp"

#include <vector>
#include <string>

struct detection
{
    int class_id;
    float confidence;
    float x, y, w, h;
};



class object_detector {
public:
    virtual std::vector<detection> detect(const yolo_image &img) {
        return std::vector<detection>();
    };
};

class yolov3_tiny_detector : public object_detector {
public:
    struct yolov3_tiny_model model;
    yolov3_tiny_detector(const std::string &model_path);
    ~yolov3_tiny_detector();

    std::vector<detection> detect(const yolo_image &img) override;
};

class yolov8_detector : public object_detector {
public:
    struct yolov8n_model model;
    yolov8_detector(const std::string &model_path);
    ~yolov8_detector();

    std::vector<detection> detect(const yolo_image &img) override;
};