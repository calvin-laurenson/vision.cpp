#pragma once

#include "common.hpp"
#include "od.hpp"
#include <string>

struct yolov8n_model {
  struct ggml_context *ctx;
  int width = 640;
  int height = 448;

  struct conv_block m0;
  struct conv_block m1;
  struct c2f_block m2;
  struct conv_block m3;
  struct c2f_block m4;
  struct conv_block m5;
  struct c2f_block m6;
  struct conv_block m7;
  struct c2f_block m8;
  struct spff_block m9;
  struct upsample_block m10;
  struct concat_block m11;
  struct c2f_block m12;
  struct upsample_block m13;
  struct concat_block m14;
  struct c2f_block m15;
  struct conv_block m16;
  struct concat_block m17;
  struct c2f_block m18;
  struct conv_block m19;
  struct concat_block m20;
  struct c2f_block m21;
  struct detect_block m22;
};

struct yolov8_detection {
    float x, y, w, h;
    std::vector<float> prob;
};

bool load_yolov8n(const std::string &fname, yolov8n_model &model);


void detect_yolov8n(const struct yolo_image & img, const yolov8n_model & model, float thresh);