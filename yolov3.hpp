#pragma once

#include "common.hpp"
#include "ggml.h"
#include <vector>
#include <string>
struct box {
  float x, y, w, h;
};

struct yolov3_detection {
  box bbox;
  std::vector<float> prob;
  float objectness;
};
struct yolov3_tiny_model {
  int width = 416;
  int height = 416;
  std::vector<conv_block> conv_blocks;
  struct ggml_context *ctx;
};

struct yolo_layer {
  int classes = 80;
  std::vector<int> mask;
  std::vector<float> anchors;
  struct ggml_tensor *predictions;

  yolo_layer(int classes, const std::vector<int> &mask,
             const std::vector<float> &anchors, struct ggml_tensor *predictions)
      : classes(classes), mask(mask), anchors(anchors),
        predictions(predictions) {}

  int entry_index(int location, int entry) const {
    int w = predictions->ne[0];
    int h = predictions->ne[1];
    int n = location / (w * h);
    int loc = location % (w * h);
    return n * w * h * (4 + classes + 1) + entry * w * h + loc;
  }
};

bool load_yolov3_tiny(const std::string &fname, yolov3_tiny_model &model);

void detect_yolov3_tiny(const struct yolo_image &img, const yolov3_tiny_model &model,
                        float thresh, float nms_thresh);