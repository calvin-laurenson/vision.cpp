#pragma once

#include "ggml.h"

#include <vector>
#include <cstdlib>
#include <cassert>



struct detection_tensors {
    ggml_tensor * bbox;
    ggml_tensor * cls;
};



// Different activation options
enum ggml_activation {
    GGML_ACTIVATION_NONE,
    GGML_ACTIVATION_SILU,
    GGML_ACTIVATION_LEAKY_RELU,
};

struct conv2d_layer {
    struct ggml_tensor * weights;
    bool use_bias = true;
    struct ggml_tensor * biases;
    int padding = 1;
    int stride = 1;
    int dilation = 1;
    int c_out = -1;
};

struct batch_norm_layer {
    struct ggml_tensor * weights;
    bool use_bias = false;
    struct ggml_tensor * biases;
    struct ggml_tensor * rolling_mean;
    struct ggml_tensor * rolling_variance;
};

struct conv_block {
    struct conv2d_layer conv;
    bool use_bn = true;
    struct batch_norm_layer bn;
    ggml_activation activation = GGML_ACTIVATION_NONE;
};

struct bottleneck_block {
    struct conv_block cv1;
    struct conv_block cv2;
    bool add = false;
};

struct c2f_block {
    struct conv_block cv1;
    struct conv_block cv2;
    std::vector<bottleneck_block> bottleneck_blocks;
};

struct max_pool_block {
    int kernel_size;
    int stride;
    int padding;
};

struct spff_block {
    struct conv_block cv1;
    struct conv_block cv2;
    struct max_pool_block mp;
};

struct upsample_block {
    float scale_factor;
};

struct concat_block {
    int dim;
};

struct detect_conv_block {
    struct conv_block cv1;
    struct conv_block cv2;
    struct ggml_tensor * conv2d_weights;
};

struct detect_block {
    std::vector<detect_conv_block> cv2;
    std::vector<detect_conv_block> cv3;
    conv2d_layer dfl;
};

void print_shape(char * prefix, const ggml_tensor * t);
signed int get_gguf_i32(gguf_context * ctx, const char * name);
float get_gguf_f32(gguf_context * ctx, const char * name);
bool get_gguf_bool(gguf_context * ctx, const char * name);
ggml_tensor * apply_conv2d(ggml_context * ctx, ggml_tensor * input, const conv2d_layer & layer);
conv_block load_conv_block(ggml_context * ctx, gguf_context * gguf_ctx, char * prefix);
ggml_tensor * apply_conv_block(ggml_context * ctx, ggml_tensor * input, const conv_block & block);
c2f_block load_c2f_block(ggml_context * ctx, gguf_context * gguf_ctx, char * prefix);
ggml_tensor * apply_bottleneck_block(ggml_context* ctx, ggml_tensor* input, const bottleneck_block& block);
ggml_tensor * apply_c2f_block(ggml_context * ctx, ggml_tensor * input, const c2f_block & block);
spff_block load_spff_block(ggml_context * ctx, gguf_context * gguf_ctx, char * prefix);
ggml_tensor * apply_max_pool_block(ggml_context * ctx, ggml_tensor * input, const max_pool_block & block);
ggml_tensor * apply_spff_block(ggml_context * ctx, ggml_tensor * input, const spff_block & block);
upsample_block load_upsample_block(gguf_context * ctx, char * prefix);
concat_block load_concat_block(gguf_context * ctx, char * prefix);