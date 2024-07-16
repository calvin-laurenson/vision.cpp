#include "common.hpp"
#include "ggml.h"

#include <vector>
#include <cstdlib>


void print_shape(char * prefix, const ggml_tensor * t)
{
    printf("%s shape:  %3d x %3d x %4d x %3d\n", prefix, (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);
}

signed int get_gguf_i32(gguf_context * ctx, const char * name)
{
    int key_id = gguf_find_key(ctx, name);
    if (key_id < 0) {
        fprintf(stderr, "Key '%s' not found\n", name);
        exit(1);
    }
    return gguf_get_val_i32(ctx, key_id);
}

float get_gguf_f32(gguf_context * ctx, const char * name)
{
    int key_id = gguf_find_key(ctx, name);
    if (key_id < 0) {
        fprintf(stderr, "Key '%s' not found\n", name);
        exit(1);
    }
    return gguf_get_val_f32(ctx, key_id);
}

bool get_gguf_bool(gguf_context * ctx, const char * name)
{
    int key_id = gguf_find_key(ctx, name);
    if (key_id < 0) {
        fprintf(stderr, "Key '%s' not found\n", name);
        exit(1);
    }
    return gguf_get_val_bool(ctx, key_id);
}

ggml_tensor * apply_conv2d(ggml_context * ctx, ggml_tensor * input, const conv2d_layer & layer)
{
    struct ggml_tensor * result = ggml_conv_2d(ctx, layer.weights, input, layer.stride, layer.stride, layer.padding, layer.padding, layer.dilation, layer.dilation);
    
    return result;
}

conv_block load_conv_block(ggml_context * ctx, gguf_context * gguf_ctx, char * prefix) {
    struct conv_block block;
    char name[256];
    
    snprintf(name, sizeof(name), "%s_conv2d_padding", prefix);
    block.conv.padding = get_gguf_i32(gguf_ctx, name);
    snprintf(name, sizeof(name), "%s_conv2d_stride", prefix);
    block.conv.stride = get_gguf_i32(gguf_ctx, name);
    snprintf(name, sizeof(name), "%s_conv2d_dilation", prefix);
    block.conv.dilation = get_gguf_i32(gguf_ctx, name);
    snprintf(name, sizeof(name), "%s_conv2d_channels_out", prefix);
    block.conv.c_out = get_gguf_i32(gguf_ctx, name);

    // printf("Loaded conv kv\n");

    snprintf(name, sizeof(name), "%s_conv2d_weights", prefix);
    block.conv.weights = ggml_get_tensor(ctx, name);
    snprintf(name, sizeof(name), "%s_conv2d_biases", prefix);
    block.conv.biases = ggml_get_tensor(ctx, name);
    block.conv.use_bias = false;
    snprintf(name, sizeof(name), "%s_bn_weights", prefix);
    block.bn.weights = ggml_get_tensor(ctx, name);
    snprintf(name, sizeof(name), "%s_bn_biases", prefix);
    block.bn.biases = ggml_get_tensor(ctx, name);
    snprintf(name, sizeof(name), "%s_bn_rolling_mean", prefix);
    block.bn.rolling_mean = ggml_get_tensor(ctx, name);
    snprintf(name, sizeof(name), "%s_bn_rolling_variance", prefix);
    block.bn.rolling_variance = ggml_get_tensor(ctx, name);

    block.activation = GGML_ACTIVATION_SILU;

    return block;
}


ggml_tensor * apply_conv_block(ggml_context * ctx, ggml_tensor * input, const conv_block & block) 
{
    print_shape("conv: input", input);
    print_shape("conv: conv2d weights", block.conv.weights);
    struct ggml_tensor * result = apply_conv2d(ctx, input, block.conv);
    print_shape("conv: conv2d output", result);
    if (block.use_bn) {
        print_shape("conv: bn rolling mean", block.bn.rolling_mean);
        result = ggml_sub(ctx, result, ggml_repeat(ctx, block.bn.rolling_mean, result));
        print_shape("conv: bn rolling mean output", result);
        print_shape("conv: bn rolling variance", block.bn.rolling_variance);
        result = ggml_div(ctx, result, ggml_sqrt(ctx, ggml_repeat(ctx, block.bn.rolling_variance, result)));
        print_shape("conv: bn weights", block.bn.weights);
        result = ggml_mul(ctx, result, ggml_repeat(ctx, block.bn.weights, result));
        if (block.bn.use_bias) {
            print_shape("conv: bn biases", block.bn.biases);
            result = ggml_add(ctx, result, ggml_repeat(ctx, block.bn.biases, result));
        }
    }
    if (block.conv.use_bias) {
        print_shape("conv: conv2d biases", block.conv.biases);
        result = ggml_add(ctx, result, ggml_repeat(ctx, block.conv.biases, result));
    }
    printf("Applied conv block\n");
    switch (block.activation)
    {
    case GGML_ACTIVATION_LEAKY_RELU:
        result = ggml_leaky_relu(ctx, result, 0.1f, true);
        break;
    case GGML_ACTIVATION_SILU:
        result = ggml_silu_inplace(ctx, result);
        break;
    default:
        break;
    }

    return result;
}

c2f_block load_c2f_block(ggml_context * ctx, gguf_context * gguf_ctx, char * prefix) {
    struct c2f_block block;
    char name[256];
    
    snprintf(name, sizeof(name), "%s_conv1", prefix);
    block.cv1 = load_conv_block(ctx, gguf_ctx, name);
    snprintf(name, sizeof(name), "%s_conv2", prefix);
    block.cv2 = load_conv_block(ctx, gguf_ctx, name);

    snprintf(name, sizeof(name), "%s_b_len", prefix);
    int bottleneck_len = get_gguf_i32(gguf_ctx, name);
    block.bottleneck_blocks.resize(bottleneck_len);
    for (int i = 0; i < bottleneck_len; i++) {
        snprintf(name, sizeof(name), "%s_b%d_conv1", prefix, i);
        block.bottleneck_blocks[i].cv1 = load_conv_block(ctx, gguf_ctx, name);
        snprintf(name, sizeof(name), "%s_b%d_conv2", prefix, i);
        block.bottleneck_blocks[i].cv2 = load_conv_block(ctx, gguf_ctx, name);
        snprintf(name, sizeof(name), "%s_b%d_add", prefix, i);
        block.bottleneck_blocks[i].add = get_gguf_bool(gguf_ctx, name);
    }
    return block;
}

ggml_tensor * apply_bottleneck_block(ggml_context* ctx, ggml_tensor* input, const bottleneck_block& block) {
    ggml_tensor* cv1_out = apply_conv_block(ctx, input, block.cv1);
    ggml_tensor* cv2_out = apply_conv_block(ctx, cv1_out, block.cv2);
    if (block.add) {
        return ggml_add(ctx, input, cv2_out);
    } else {
        return cv2_out;
    }
}

ggml_tensor* apply_c2f_block(ggml_context* ctx, ggml_tensor* input, const c2f_block& block) {
    // Apply cv1 convolution
    ggml_tensor* cv1_out = apply_conv_block(ctx, input, block.cv1);
    print_shape(0, cv1_out);

    int64_t w = cv1_out->ne[0];  // width = 160
    int64_t h = cv1_out->ne[1];  // height = 160
    int64_t c = cv1_out->ne[2];  // channels = 32
    int64_t n = cv1_out->ne[3];  // batch size = 1
    int64_t c_half = c / 2;

    // Create views for the first and second half of the channels
    struct ggml_tensor * y1 = ggml_view_4d(ctx, cv1_out, w, h, c_half, n, 
                                       cv1_out->nb[1], cv1_out->nb[2], cv1_out->nb[3], 0);

    struct ggml_tensor * y2 = ggml_view_4d(ctx, cv1_out, w, h, c_half, n, 
                                        cv1_out->nb[1], cv1_out->nb[2], cv1_out->nb[3], 
                                        c_half * cv1_out->nb[2]);
    print_shape("c2f: y1", y1);
    print_shape("c2f: y2", y2);
    
    // Apply Bottleneck layers
    std::vector<ggml_tensor*> cat_tensors = {y1, y2};
    ggml_tensor* y_last = y2;
    for (const auto& bottleneck : block.bottleneck_blocks) {
        y_last = apply_bottleneck_block(ctx, y_last, bottleneck);
        cat_tensors.push_back(y_last);
    }
    print_shape("c2f: y_last final", y_last);

    // Concatenate all tensors
    ggml_tensor* cat_out = cat_tensors[0];
    print_shape("c2f: cat original", cat_out);
    for (size_t i = 1; i < cat_tensors.size(); ++i) {
        cat_out = ggml_concat(ctx, cat_out, cat_tensors[i], 2);  // Concatenate along dimension 1 (channels)
        print_shape("c2f: catted", cat_out);
    }
    print_shape("c2f: final-cat", cat_out);
    print_shape("ctf: cv2-weights", block.cv2.conv.weights);
    
    // Apply cv2 convolution
    ggml_tensor* cv2_out = apply_conv_block(ctx, cat_out, block.cv2);

    return cv2_out;
}


spff_block load_spff_block(ggml_context * ctx, gguf_context * gguf_ctx, char * prefix) {
    struct spff_block block;
    char name[256];
    
    snprintf(name, sizeof(name), "%s_conv1", prefix);
    block.cv1 = load_conv_block(ctx, gguf_ctx, name);
    snprintf(name, sizeof(name), "%s_conv2", prefix);
    block.cv2 = load_conv_block(ctx, gguf_ctx, name);

    snprintf(name, sizeof(name), "%s_mp_kernel_size", prefix);
    block.mp.kernel_size = get_gguf_i32(gguf_ctx, name);
    snprintf(name, sizeof(name), "%s_mp_stride", prefix);
    block.mp.stride = get_gguf_i32(gguf_ctx, name);
    snprintf(name, sizeof(name), "%s_mp_padding", prefix);
    block.mp.padding = get_gguf_i32(gguf_ctx, name);

    return block;
}

ggml_tensor * apply_max_pool_block(ggml_context * ctx, ggml_tensor * input, const max_pool_block & block) {
    return ggml_pool_2d(ctx, input, GGML_OP_POOL_MAX, block.kernel_size, block.kernel_size, block.stride, block.stride, block.padding, block.padding);
}

ggml_tensor * apply_spff_block(ggml_context * ctx, ggml_tensor * input, const spff_block & block) {
        std::vector<struct ggml_tensor*> y;
        

        // Apply cv1
        struct ggml_tensor* cv1_out = apply_conv_block(ctx, input, block.cv1);
        print_shape("spff: cv1_out", cv1_out);
        y.push_back(cv1_out);

        // Apply maxpool 3 times
        for (int i = 0; i < 3; i++) {
            struct ggml_tensor* pooled = apply_max_pool_block(ctx, cv1_out, block.mp);
            print_shape("spff: pooled", pooled);
            y.push_back(pooled);
        }

        // Concatenate tensors
        ggml_tensor* cat = y[0];

        for (size_t i = 1; i < y.size(); i++) {
            print_shape("spff: cat input", y[i]);
            cat = ggml_concat(ctx, cat, y[i], 2);
            print_shape("spff: catted", cat);
        }
        print_shape("spff: final cat", cat);
        // Apply cv2
        struct ggml_tensor* output = apply_conv_block(ctx, cat, block.cv2);

        return output;
}

upsample_block load_upsample_block(gguf_context * ctx, char * prefix) {
    struct upsample_block block;
    char name[256];
    
    snprintf(name, sizeof(name), "%s_scale_factor", prefix);
    block.scale_factor = get_gguf_f32(ctx, name);

    return block;
}

concat_block load_concat_block(gguf_context * ctx, char * prefix) {
    struct concat_block block;
    char name[256];
    
    snprintf(name, sizeof(name), "%s_dim", prefix);
    block.dim = get_gguf_i32(ctx, name);
    return block;
}





