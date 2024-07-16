#pragma once
#include "yolov3.hpp"
#include "yolov8.hpp"

#include <cassert>
#include <vector>
#include <string>

struct detection
{
    int class_id;
    float confidence;
    float x, y, w, h;
};

struct yolo_image {
    int w, h, c;
    std::vector<float> data;

    yolo_image() : w(0), h(0), c(0) {}
    yolo_image(int w, int h, int c) : w(w), h(h), c(c), data(w*h*c) {}

    float get_pixel(int x, int y, int c) const {
        assert(x >= 0 && x < w && y >= 0 && y < h && c >= 0 && c < this->c);
        return data[c*w*h + y*w + x];
    }

    void set_pixel(int x, int y, int c, float val) {
        assert(x >= 0 && x < w && y >= 0 && y < h && c >= 0 && c < this->c);
        data[c*w*h + y*w + x] = val;
    }

    void add_pixel(int x, int y, int c, float val) {
        assert(x >= 0 && x < w && y >= 0 && y < h && c >= 0 && c < this->c);
        data[c*w*h + y*w + x] += val;
    }

    void fill(float val) {
        std::fill(data.begin(), data.end(), val);
    }
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