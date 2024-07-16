#include "common.hpp"
#include "od.hpp"
#include "yolov8.hpp"

#include <string>
#include <cstring>



ggml_tensor *concat_yolov8_detections(ggml_context *ctx,
                                      std::vector<ggml_tensor *> outputs) {
  int64_t no = 1;
  int64_t bs = 1;
  std::vector<ggml_tensor *> reshaped;
  reshaped.resize(outputs.size());

  for (int i = 0; i < outputs.size(); i++) {
    reshaped[i] = ggml_reshape_3d(ctx, outputs[i], bs, no,
                                  ggml_nelements(outputs[i]) / (no * bs));
    print_shape("view", reshaped[i]);
  }

  ggml_tensor *cat = reshaped[0];
  for (int i = 1; i < outputs.size(); i++) {
    cat = ggml_concat(ctx, cat, reshaped[i], 2);
  }

  return cat;
}

static ggml_tensor *apply_dfl(ggml_context *ctx, ggml_tensor *input,
                              ggml_tensor *weights) {
  int64_t b = input->ne[0];
  int64_t a = input->ne[2];

  // 1. Reshape x
  ggml_tensor *reshaped = ggml_reshape_4d(ctx, input, a, 16, 4, b);

  print_shape("dfl: reshaped", reshaped);

  // 2. Transpose dimensions 2 and 1. New shape is [a, 4, 16, b]
  ggml_tensor *permuted = ggml_permute(ctx, reshaped, 0, 2, 1, 3);
  permuted = ggml_cont(ctx, permuted);
  print_shape("dfl: permuted", permuted);

  // 3. Apply softmax along dimension 1
  ggml_tensor *softmaxed = ggml_soft_max(ctx, permuted);

  print_shape("dfl: softmaxed", softmaxed);
  print_shape("dfl: weights", weights);
  // 4. Apply convolution (assuming 'self.conv' is defined elsewhere)
  ggml_tensor *conved = ggml_conv_2d(ctx, weights, softmaxed, 1, 1, 0, 0, 1, 1);

  print_shape("dfl: conved", conved);

  // 5. Reshape the result back to 3d
  return ggml_reshape_3d(ctx, conved, b, 4, a);
}

detect_conv_block load_detect_conv_block(ggml_context *ctx,
                                         gguf_context *gguf_ctx, char *prefix) {
  struct detect_conv_block block;
  char name[256];

  snprintf(name, sizeof(name), "%s_conv1", prefix);
  block.cv1 = load_conv_block(ctx, gguf_ctx, name);
  snprintf(name, sizeof(name), "%s_conv2", prefix);
  block.cv2 = load_conv_block(ctx, gguf_ctx, name);
  snprintf(name, sizeof(name), "%s_conv2d_weights", prefix);
  block.conv2d_weights = ggml_get_tensor(ctx, name);

  return block;
}

ggml_tensor *apply_detect_conv_block(ggml_context *ctx, ggml_tensor *input,
                                     const detect_conv_block &block) {
  ggml_tensor *cv1_out = apply_conv_block(ctx, input, block.cv1);
  ggml_tensor *cv2_out = apply_conv_block(ctx, cv1_out, block.cv2);
  return ggml_conv_2d(ctx, block.conv2d_weights, cv2_out, 1, 1, 0, 0, 1, 1);
}

static detect_block load_detect_block(ggml_context * ctx, gguf_context * gguf_ctx, char * prefix) {
    struct detect_block block;
    char name[256];
    
    snprintf(name, sizeof(name), "%s_conv2_len", prefix);
    int cv2_len = get_gguf_i32(gguf_ctx, name);
    block.cv2.resize(cv2_len);
    for (int i = 0; i < cv2_len; i++) {
        sprintf(name, "%s_conv2_m%d", prefix, i);
        block.cv2[i] = load_detect_conv_block(ctx, gguf_ctx, name);
    }
    // printf("Loaded detect cv2\n");
    snprintf(name, sizeof(name), "%s_conv3_len", prefix);
    int cv3_len = get_gguf_i32(gguf_ctx, name);
    block.cv3.resize(cv3_len);
    for (int i = 0; i < cv3_len; i++) {
        sprintf(name, "%s_conv3_m%d", prefix, i);
        block.cv3[i] = load_detect_conv_block(ctx, gguf_ctx, name);
    }
    // printf("Loaded detect cv3\n");

    snprintf(name, sizeof(name), "%s_dfl_weights", prefix);
    block.dfl.weights = ggml_get_tensor(ctx, name);
    // printf("Loaded detect dfl_weights\n");
    block.dfl.use_bias = false;

    return block;
}
static detection_tensors apply_detect_block(ggml_context *ctx,
                                            std::vector<ggml_tensor *> inputs,
                                            const detect_block &block) {
  std::vector<ggml_tensor *> x;

  for (size_t i = 0; i < inputs.size(); i++) {
    ggml_tensor *cv2_out =
        apply_detect_conv_block(ctx, inputs[i], block.cv2[i]);
    ggml_tensor *cv3_out =
        apply_detect_conv_block(ctx, inputs[i], block.cv3[i]);
    ggml_tensor *catted = ggml_concat(ctx, cv2_out, cv3_out, 2);
    print_shape("detect: catted", catted);
    x.push_back(catted);
  }
  int64_t nc = 80;
  int64_t reg_max = 16;
  int64_t bs = 1;
  int64_t no = nc + reg_max * 4;
  std::vector<ggml_tensor *> reshaped;
  reshaped.resize(x.size());

  for (int i = 0; i < x.size(); i++) {
    reshaped[i] =
        ggml_reshape_3d(ctx, x[i], bs, no, ggml_nelements(x[i]) / (no * bs));
    print_shape("view", reshaped[i]);
  }

  ggml_tensor *cat = reshaped[0];
  for (int i = 1; i < x.size(); i++) {
    cat = ggml_concat(ctx, cat, reshaped[i], 2);
  }

  ggml_tensor *box =
      ggml_view_3d(ctx, cat, 1, 64, 5880, cat->nb[1], cat->nb[2], 0);
  box = ggml_cont(ctx, box);
  print_shape("detect: box", box);

  ggml_tensor *cls = ggml_view_3d(ctx, cat, 1, 80, 5880, cat->nb[1], cat->nb[2],
                                  64 * cat->nb[2]);
  cls = ggml_cont(ctx, cls);
  cls = ggml_sigmoid(ctx, cls);
  print_shape("detect: cls", cls);

  // Box gets run through dfl
  ggml_tensor *box_out = apply_dfl(ctx, box, block.dfl.weights);
  box_out = ggml_cont(ctx, box_out);
  print_shape("detect: box_out", box_out);

  return {box_out, cls};
}

bool load_yolov8n(const std::string &fname, yolov8n_model &model) {
  struct gguf_init_params params = {
      /*.no_alloc   =*/false,
      /*.ctx        =*/&model.ctx,
  };
  gguf_context *ctx = gguf_init_from_file(fname.c_str(), params);
  if (!ctx) {
    fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
    return false;
  }

  printf("Model has %d keys\n", gguf_get_n_kv(ctx));

  model.width = 640;
  model.height = 448;

  model.m0 = load_conv_block(model.ctx, ctx, "m0");
  printf("m0 loaded\n");
  model.m1 = load_conv_block(model.ctx, ctx, "m1");
  printf("m1 loaded\n");
  model.m2 = load_c2f_block(model.ctx, ctx, "m2");
  printf("m2 loaded\n");
  model.m3 = load_conv_block(model.ctx, ctx, "m3");
  printf("m3 loaded\n");
  model.m4 = load_c2f_block(model.ctx, ctx, "m4");
  printf("m4 loaded\n");
  model.m5 = load_conv_block(model.ctx, ctx, "m5");
  printf("m5 loaded\n");
  model.m6 = load_c2f_block(model.ctx, ctx, "m6");
  printf("m6 loaded\n");
  model.m7 = load_conv_block(model.ctx, ctx, "m7");
  printf("m7 loaded\n");
  model.m8 = load_c2f_block(model.ctx, ctx, "m8");
  printf("m8 loaded\n");
  model.m9 = load_spff_block(model.ctx, ctx, "m9");
  printf("m9 loaded\n");
  model.m10 = load_upsample_block(ctx, "m10");
  printf("m10 loaded\n");
  model.m11 = load_concat_block(ctx, "m11");
  printf("m11 loaded\n");
  model.m12 = load_c2f_block(model.ctx, ctx, "m12");
  printf("m12 loaded\n");
  model.m13 = load_upsample_block(ctx, "m13");
  printf("m13 loaded\n");
  model.m14 = load_concat_block(ctx, "m14");
  printf("m14 loaded\n");
  model.m15 = load_c2f_block(model.ctx, ctx, "m15");
  printf("m15 loaded\n");
  model.m16 = load_conv_block(model.ctx, ctx, "m16");
  printf("m16 loaded\n");
  model.m17 = load_concat_block(ctx, "m17");
  printf("m17 loaded\n");
  model.m18 = load_c2f_block(model.ctx, ctx, "m18");
  printf("m18 loaded\n");
  model.m19 = load_conv_block(model.ctx, ctx, "m19");
  printf("m19 loaded\n");
  model.m20 = load_concat_block(ctx, "m20");
  printf("m20 loaded\n");
  model.m21 = load_c2f_block(model.ctx, ctx, "m21");
  printf("m21 loaded\n");
  model.m22 = load_detect_block(model.ctx, ctx, "m22");
  printf("m22 loaded\n");

  return true;
}

void parse_yolov8_detections(ggml_context * ctx, detection_tensors &tensors, std::vector<yolov8_detection> &detections) {  
    // Bbox is [1, 4, 5880]
    float * bbox = ggml_get_data_f32(tensors.bbox);
    printf("bbox: %f %f %f %f\n", bbox[0], bbox[1], bbox[2], bbox[3]);
    // Cls is [1, 80, 5880]
    float * cls = ggml_get_data_f32(tensors.cls);

    int bs = tensors.bbox->ne[0];
    int no = tensors.bbox->ne[2];
    int nc = 80;
    int reg_max = 16;
    int stride = 4;
    int n = 3;
    int w = 640;
    int h = 448;

    
    for (int i = 0; i < no; i++) {
        int index = i*stride*4;
        yolov8_detection det;
        det.prob.resize(nc);
        // Yolo outputs in ltrb so we need to convert to xywh
        float l = bbox[index + 0];
        float t = bbox[index + 1];
        float r = bbox[index + 2];
        float b = bbox[index + 3];
        // printf("bbox: %f %f %f %f\n", l, t, r, b);
        det.x = (l + r) / 2;
        det.y = (t + b) / 2;
        det.w = r - l;
        det.h = b - t;

        for (int k = 0; k < nc; k++) {
            det.prob[k] = cls[no*nc + i*nc + k];
            if (det.prob[k] > 0.5) {
                printf("cls: %d %f\n", k, det.prob[k]);
            }
        }

        detections.push_back(det);

    }
    
}

void detect_yolov8n(const yolo_image & img, const yolov8n_model & model, float thresh)
{
    static size_t buf_size = 50000000 * sizeof(float) * 4;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    std::vector<yolov8_detection> detections;

    struct ggml_tensor * input = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, model.width, model.height, 3, 1);
    std::memcpy(input->data, img.data.data(), ggml_nbytes(input));
    // input = ggml_reshape_4d(ctx0, input, 1, 3, model.width, model.height);
    ggml_set_name(input, "input");

    printf("Input shape:  %3d x %3d x %4d x %3d\n", (int)input->ne[0], (int)input->ne[1], (int)input->ne[2], (int)input->ne[3]);
    printf("Beginning to build graph...\n");

    struct ggml_tensor * result = apply_conv_block(ctx0, input, model.m0);
    print_shape("Layer 0", result);
    result = apply_conv_block(ctx0, result, model.m1);
    print_shape("Layer 1", result);
    result = apply_c2f_block(ctx0, result, model.m2);
    print_shape("Layer 2", result);
    result = apply_conv_block(ctx0, result, model.m3);
    print_shape("Layer 3", result);
    struct ggml_tensor * m4 = apply_c2f_block(ctx0, result, model.m4);
    print_shape("Layer 4", m4);
    result = apply_conv_block(ctx0, m4, model.m5);
    print_shape("Layer 5", result);
    struct ggml_tensor * m6 = apply_c2f_block(ctx0, result, model.m6);
    print_shape("Layer 6", m6);
    result = apply_conv_block(ctx0, m6, model.m7);
    print_shape("Layer 7", result);
    result = apply_c2f_block(ctx0, result, model.m8);
    print_shape("Layer 8", result);
    struct ggml_tensor * m9 = apply_spff_block(ctx0, result, model.m9);
    print_shape("Layer 9", m9);
    result = ggml_upscale(ctx0, m9, 2);
    print_shape("Layer 10", result);
    result = ggml_concat(ctx0, result, m6, 2);
    print_shape("Layer 11", result);
    struct ggml_tensor * m12 = apply_c2f_block(ctx0, result, model.m12);
    print_shape("Layer 12", m12);
    result = ggml_upscale(ctx0, m12, 2);
    print_shape("Layer 13", result);
    result = ggml_concat(ctx0, result, m4, 2);
    print_shape("Layer 14", result);
    struct ggml_tensor * m15 = apply_c2f_block(ctx0, result, model.m15);
    print_shape("Layer 15", m15);
    result = apply_conv_block(ctx0, m15, model.m16);
    print_shape("Layer 16", result);
    result = ggml_concat(ctx0, result, m12, 2);
    print_shape("Layer 17", result);
    struct ggml_tensor * m18 = apply_c2f_block(ctx0, result, model.m18);
    print_shape("Layer 18", m18);
    result = apply_conv_block(ctx0, m18, model.m19);
    print_shape("Layer 19", result);
    result = ggml_concat(ctx0, result, m9, 2);
    print_shape("Layer 20", result);
    struct ggml_tensor * m21 = apply_c2f_block(ctx0, result, model.m21);
    print_shape("Layer 21", m21);
    std::vector<ggml_tensor*> inputs = {m15, m18, m21};
    detection_tensors detected = apply_detect_block(ctx0, inputs, model.m22);
    print_shape("bbox", detected.bbox);
    print_shape("cls", detected.cls);
    
    // ggml_graph_dump_dot(gf, gf, "graph.dot");
    ggml_build_forward_expand(gf, detected.bbox);
    ggml_build_forward_expand(gf, detected.cls);

    ggml_graph_dump_dot(gf, NULL, "graph.dot");
    
    printf("Graph built\n");
    ggml_graph_compute_with_ctx(ctx0, gf, 1);
    printf("Graph computed\n");
    parse_yolov8_detections(ctx0, detected, detections);
    printf("Detected %d objects before nms\n", (int)detections.size());
    // std::vector<simple_detection> nms_dets = do_nms_sort(detections, 80, .45);
    // printf("Detected %d objects after nms\n", (int)nms_dets.size());
    printf("Drawing detections...\n");
    ggml_free(ctx0);


    
}
