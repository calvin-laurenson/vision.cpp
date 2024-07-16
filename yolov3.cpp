#include "yolov3.hpp"
#include "od.hpp"
#include "common.hpp"

#include <algorithm>
#include <complex>
#include <cstring>
#include <string>
#include <vector>



static void print_layer_shape(int layer, const ggml_tensor *t) {
  printf("Layer %2d output shape:  %3d x %3d x %4d x %3d\n", layer,
         (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);
}

static float overlap(float x1, float w1, float x2, float w2) {
  float l1 = x1 - w1 / 2;
  float l2 = x2 - w2 / 2;
  float left = l1 > l2 ? l1 : l2;
  float r1 = x1 + w1 / 2;
  float r2 = x2 + w2 / 2;
  float right = r1 < r2 ? r1 : r2;
  return right - left;
}

static float box_intersection(const box &a, const box &b) {
  float w = overlap(a.x, a.w, b.x, b.w);
  float h = overlap(a.y, a.h, b.y, b.h);
  if (w < 0 || h < 0)
    return 0;
  float area = w * h;
  return area;
}

static float box_union(const box &a, const box &b) {
  float i = box_intersection(a, b);
  float u = a.w * a.h + b.w * b.h - i;
  return u;
}

static float box_iou(const box &a, const box &b) {
  return box_intersection(a, b) / box_union(a, b);
}

static void do_nms_sort(std::vector<yolov3_detection> &dets, int classes,
                        float thresh) {
  int k = (int)dets.size() - 1;
  for (int i = 0; i <= k; ++i) {
    if (dets[i].objectness == 0) {
      std::swap(dets[i], dets[k]);
      --k;
      --i;
    }
  }
  int total = k + 1;
  for (int k = 0; k < classes; ++k) {
    std::sort(dets.begin(), dets.begin() + total,
              [=](const yolov3_detection &a, const yolov3_detection &b) {
                return a.prob[k] > b.prob[k];
              });
    for (int i = 0; i < total; ++i) {
      if (dets[i].prob[k] == 0) {
        continue;
      }
      box a = dets[i].bbox;
      for (int j = i + 1; j < total; ++j) {
        box b = dets[j].bbox;
        if (box_iou(a, b) > thresh) {
          dets[j].prob[k] = 0;
        }
      }
    }
  }
}



bool load_yolov3_tiny(const std::string &fname, yolov3_tiny_model &model) {
  struct gguf_init_params params = {
      /*.no_alloc   =*/false,
      /*.ctx        =*/&model.ctx,
  };
  gguf_context *ctx = gguf_init_from_file(fname.c_str(), params);
  if (!ctx) {
    fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
    return false;
  }
  model.width = 416;
  model.height = 416;
  model.conv_blocks.resize(13);
  model.conv_blocks[7].conv.padding = 0;
  model.conv_blocks[9].conv.padding = 0;
  model.conv_blocks[9].use_bn = false;
  model.conv_blocks[10].conv.padding = 0;
  model.conv_blocks[12].conv.padding = 0;
  model.conv_blocks[12].use_bn = false;

  for (int i = 0; i < (int)model.conv_blocks.size(); i++) {
    char name[256];
    snprintf(name, sizeof(name), "l%d_weights", i);
    model.conv_blocks[i].conv.weights = ggml_get_tensor(model.ctx, name);
    snprintf(name, sizeof(name), "l%d_biases", i);
    model.conv_blocks[i].conv.biases = ggml_get_tensor(model.ctx, name);
    if (model.conv_blocks[i].use_bn) {
      snprintf(name, sizeof(name), "l%d_scales", i);
      model.conv_blocks[i].bn.weights = ggml_get_tensor(model.ctx, name);
      snprintf(name, sizeof(name), "l%d_rolling_mean", i);
      model.conv_blocks[i].bn.rolling_mean = ggml_get_tensor(model.ctx, name);
      snprintf(name, sizeof(name), "l%d_rolling_variance", i);
      model.conv_blocks[i].bn.rolling_variance =
          ggml_get_tensor(model.ctx, name);
    }
    model.conv_blocks[i].activation = GGML_ACTIVATION_LEAKY_RELU;
  }

  model.conv_blocks[9].activation = GGML_ACTIVATION_NONE;
  model.conv_blocks[12].activation = GGML_ACTIVATION_NONE;

  return true;
}

static void activate_array(float *x, const int n) {
  // logistic activation
  for (int i = 0; i < n; i++) {
    x[i] = 1. / (1. + exp(-x[i]));
  }
}

static void apply_yolo(yolo_layer &layer) {
  int w = layer.predictions->ne[0];
  int h = layer.predictions->ne[1];
  int N = layer.mask.size();
  float *data = ggml_get_data_f32(layer.predictions);
  for (int n = 0; n < N; n++) {
    int index = layer.entry_index(n * w * h, 0);
    activate_array(data + index, 2 * w * h);
    index = layer.entry_index(n * w * h, 4);
    activate_array(data + index, (1 + layer.classes) * w * h);
  }
}
static box get_yolo_box(const yolo_layer &layer, int n, int index, int i, int j,
                        int lw, int lh, int w, int h, int stride) {
  float *predictions = ggml_get_data_f32(layer.predictions);
  box b;
  b.x = (i + predictions[index + 0 * stride]) / lw;
  b.y = (j + predictions[index + 1 * stride]) / lh;
  b.w = exp(predictions[index + 2 * stride]) * layer.anchors[2 * n] / w;
  b.h = exp(predictions[index + 3 * stride]) * layer.anchors[2 * n + 1] / h;
  return b;
}

static void correct_yolo_box(box &b, int im_w, int im_h, int net_w, int net_h) {
  int new_w = 0;
  int new_h = 0;
  if (((float)net_w / im_w) < ((float)net_h / im_h)) {
    new_w = net_w;
    new_h = (im_h * net_w) / im_w;
  } else {
    new_h = net_h;
    new_w = (im_w * net_h) / im_h;
  }
  b.x = (b.x - (net_w - new_w) / 2. / net_w) / ((float)new_w / net_w);
  b.y = (b.y - (net_h - new_h) / 2. / net_h) / ((float)new_h / net_h);
  b.w *= (float)net_w / new_w;
  b.h *= (float)net_h / new_h;
}
static void get_yolo_detections(const yolo_layer &layer,
                                std::vector<yolov3_detection> &detections, int im_w,
                                int im_h, int netw, int neth, float thresh) {
  int w = layer.predictions->ne[0];
  int h = layer.predictions->ne[1];
  int N = layer.mask.size();
  float *predictions = ggml_get_data_f32(layer.predictions);
  std::vector<yolov3_detection> result;
  for (int i = 0; i < w * h; i++) {
    for (int n = 0; n < N; n++) {
      int obj_index = layer.entry_index(n * w * h + i, 4);
      float objectness = predictions[obj_index];
      // printf("i=%d, n=%d, objectness=%.2f\n", i, n, objectness);
      if (objectness <= thresh) {
        continue;
      }
      yolov3_detection det;
      int box_index = layer.entry_index(n * w * h + i, 0);
      int row = i / w;
      int col = i % w;
      det.bbox = get_yolo_box(layer, layer.mask[n], box_index, col, row, w, h,
                              netw, neth, w * h);
      correct_yolo_box(det.bbox, im_w, im_h, netw, neth);
      det.objectness = objectness;
      det.prob.resize(layer.classes);
      for (int j = 0; j < layer.classes; j++) {
        int class_index = layer.entry_index(n * w * h + i, 4 + 1 + j);
        float prob = objectness * predictions[class_index];
        det.prob[j] = (prob > thresh) ? prob : 0;
      }
      detections.push_back(det);
    }
  }
}

void detect_yolov3_tiny(const yolo_image &img, const yolov3_tiny_model &model,
                        float thresh, float nms_thresh) {
  static size_t buf_size = 20000000 * sizeof(float) * 4;
  static void *buf = malloc(buf_size);

  struct ggml_init_params params = {
      /*.mem_size   =*/buf_size,
      /*.mem_buffer =*/buf,
      /*.no_alloc   =*/false,
  };

  struct ggml_context *ctx0 = ggml_init(params);
  struct ggml_cgraph *gf = ggml_new_graph(ctx0);
  std::vector<yolov3_detection> detections;
  assert(img.w == model.width && img.h == model.height);
  struct ggml_tensor *input =
      ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, model.width, model.height, 3, 1);
  std::memcpy(input->data, img.data.data(), ggml_nbytes(input));
  ggml_set_name(input, "input");

  struct ggml_tensor *result =
      apply_conv_block(ctx0, input, model.conv_blocks[0]);
  print_layer_shape(0, result);
  result = ggml_pool_2d(ctx0, result, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
  print_layer_shape(1, result);
  result = apply_conv_block(ctx0, result, model.conv_blocks[1]);
  print_layer_shape(2, result);
  result = ggml_pool_2d(ctx0, result, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
  print_layer_shape(3, result);
  result = apply_conv_block(ctx0, result, model.conv_blocks[2]);
  print_layer_shape(4, result);
  result = ggml_pool_2d(ctx0, result, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
  print_layer_shape(5, result);
  result = apply_conv_block(ctx0, result, model.conv_blocks[3]);
  print_layer_shape(6, result);
  result = ggml_pool_2d(ctx0, result, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
  print_layer_shape(7, result);
  result = apply_conv_block(ctx0, result, model.conv_blocks[4]);
  struct ggml_tensor *layer_8 = result;
  print_layer_shape(8, result);
  result = ggml_pool_2d(ctx0, result, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
  print_layer_shape(9, result);
  result = apply_conv_block(ctx0, result, model.conv_blocks[5]);
  print_layer_shape(10, result);
  result = ggml_pool_2d(ctx0, result, GGML_OP_POOL_MAX, 2, 2, 1, 1, 0.5, 0.5);
  print_layer_shape(11, result);
  result = apply_conv_block(ctx0, result, model.conv_blocks[6]);
  print_layer_shape(12, result);
  result = apply_conv_block(ctx0, result, model.conv_blocks[7]);
  struct ggml_tensor *layer_13 = result;
  print_layer_shape(13, result);
  result = apply_conv_block(ctx0, result, model.conv_blocks[8]);
  print_layer_shape(14, result);
  result = apply_conv_block(ctx0, result, model.conv_blocks[9]);
  struct ggml_tensor *layer_15 = result;
  print_layer_shape(15, result);
  result = apply_conv_block(ctx0, layer_13, model.conv_blocks[10]);
  print_layer_shape(18, result);
  result = ggml_upscale(ctx0, result, 2);
  print_layer_shape(19, result);
  result = ggml_concat(ctx0, result, layer_8, 2);
  print_layer_shape(20, result);
  result = apply_conv_block(ctx0, result, model.conv_blocks[11]);
  print_layer_shape(21, result);
  result = apply_conv_block(ctx0, result, model.conv_blocks[12]);
  struct ggml_tensor *layer_22 = result;
  print_layer_shape(22, result);

  ggml_build_forward_expand(gf, layer_15);
  ggml_build_forward_expand(gf, layer_22);
  ggml_graph_compute_with_ctx(ctx0, gf, 1);

  yolo_layer yolo16{80,
                    {3, 4, 5},
                    {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319},
                    layer_15};
  apply_yolo(yolo16);
  get_yolo_detections(yolo16, detections, img.w, img.h, model.width,
                      model.height, thresh);

  yolo_layer yolo23{80,
                    {0, 1, 2},
                    {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319},
                    layer_22};
  apply_yolo(yolo23);
  get_yolo_detections(yolo23, detections, img.w, img.h, model.width,
                      model.height, thresh);
  do_nms_sort(detections, yolo23.classes, nms_thresh);
  // Print out detections to the console
  for (int i = 0; i < (int)detections.size(); i++) {
    yolov3_detection &det = detections[i];
    int class_id = -1;
    float prob = 0;
    for (int j = 0; j < 80; j++) {
      if (det.prob[j] > prob) {
        class_id = j;
        prob = det.prob[j];
      }
    }

    if (prob > thresh) {
      printf("%d: %.2f%%\n", class_id, prob * 100);
    }
  }

  ggml_free(ctx0);
}