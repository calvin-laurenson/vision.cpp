#include "od.hpp"
#include "ggml.h"

#include <vector>

yolov3_tiny_detector::yolov3_tiny_detector(const std::string &model_path) {
    load_yolov3_tiny(model_path, model);
}

yolov3_tiny_detector::~yolov3_tiny_detector() {
    ggml_free(model.ctx);
}

std::vector<detection> yolov3_tiny_detector::detect(const struct yolo_image &img) {
    detect_yolov3_tiny(img, model, 0.3, 0.5);
    return std::vector<detection>();
}

yolov8_detector::yolov8_detector(const std::string &model_path) {
    load_yolov8n(model_path, model);
}

yolov8_detector::~yolov8_detector() {
    ggml_free(model.ctx);
}

std::vector<detection> yolov8_detector::detect(const struct yolo_image &img) {
    detect_yolov8n(img, model, 0.3);
    return std::vector<detection>();
}