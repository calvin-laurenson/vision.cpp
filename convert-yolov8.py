import sys
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel, parse_model
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.block import C2f, SPPF, Bottleneck
from ultralytics.nn.modules.head import Detect
import gguf
import numpy as np
from typing import cast

def save_conv(writer: gguf.GGUFWriter, prefix: str, conv: Conv):
    writer.add_int32(prefix + "_conv2d_kernel_size", conv.conv.kernel_size[0])
    writer.add_int32(prefix + "_conv2d_stride", conv.conv.stride[0])
    writer.add_int32(prefix + "_conv2d_dilation", conv.conv.dilation[0])
    writer.add_int32(prefix + "_conv2d_channels_out", conv.conv.out_channels)
    states = conv.state_dict()
    writer.add_tensor(prefix + "_conv2d_weights", states["conv.weight"].detach().numpy().astype(np.float16), raw_shape=states["conv.weight"].shape)
    conv2d_padding = conv.conv.padding
    assert all([x == conv2d_padding[0] for x in conv2d_padding]), "Only symmetric padding is supported"
    writer.add_int32(prefix + "_conv2d_padding", conv2d_padding[0])
    bn_weight = states["bn.weight"].detach().numpy()
    writer.add_tensor(prefix + "_bn_weights", bn_weight.reshape((1, bn_weight.shape[0], 1, 1)), raw_shape=[1, bn_weight.shape[0], 1, 1])
    writer.add_tensor(prefix + "_bn_biases", states["bn.bias"].detach().numpy(), raw_shape=states["bn.bias"].shape)
    bn_running_mean = states["bn.running_mean"].detach().numpy()
    writer.add_tensor(prefix + "_bn_rolling_mean", bn_running_mean.reshape((1, bn_running_mean.shape[0], 1, 1)), raw_shape=[1, bn_running_mean.shape[0], 1, 1])
    bn_running_var = states["bn.running_var"].detach().numpy()
    writer.add_tensor(prefix + "_bn_rolling_variance", bn_running_var.reshape((1, bn_running_var.shape[0], 1, 1)), raw_shape=[1, bn_running_var.shape[0], 1, 1])

    assert isinstance(conv.act, torch.nn.SiLU)
    writer.add_string(prefix + "_act", "silu")

def save_c2f(writer: gguf.GGUFWriter, prefix: str, c2f: C2f):
    # print(f"{c2f.state_dict().keys()=}")
    save_conv(writer, prefix + "_conv1", c2f.cv1)
    save_conv(writer, prefix + "_conv2", c2f.cv2)
    writer.add_int32(prefix + "_b_len", len(c2f.m))
    for i, m in enumerate(c2f.m):
        m = cast(Bottleneck, m)
        save_conv(writer, prefix + f"_b{i}_conv1", m.cv1)
        save_conv(writer, prefix + f"_b{i}_conv2", m.cv2)
        writer.add_bool(prefix + f"_b{i}_add", m.add)


def save_sppf(writer: gguf.GGUFWriter, prefix: str, sppf: SPPF):
    save_conv(writer, prefix + "_conv1", sppf.cv1)
    save_conv(writer, prefix + "_conv2", sppf.cv2)
    writer.add_int32(prefix + "_mp_kernel_size", sppf.m.kernel_size)
    writer.add_int32(prefix + "_mp_stride", sppf.m.stride)
    writer.add_int32(prefix + "_mp_padding", sppf.m.padding)


def save_detect_conv(writer: gguf.GGUFWriter, prefix: str, conv: torch.nn.Sequential):
    children = list(conv.children())
    # print(children)
    save_conv(writer, f"{prefix}_conv1", children[0])
    save_conv(writer, f"{prefix}_conv2", children[1])
    writer.add_tensor(f"{prefix}_conv2d_weights", children[2].state_dict()["weight"].detach().numpy().astype(np.float16), raw_shape=children[2].state_dict()["weight"].shape)

def save_detect(writer: gguf.GGUFWriter, prefix: str, detect: Detect):
    print(prefix+"_conv2_len", len(detect.cv2))
    writer.add_int32(prefix + "_conv2_len", len(detect.cv2))
    for i, m in enumerate(detect.cv2):
        save_detect_conv(writer, f"{prefix}_conv2_m{i}", m)
    writer.add_int32(prefix + "_conv3_len", len(detect.cv3))
    for i, m in enumerate(detect.cv3):
        save_detect_conv(writer, f"{prefix}_conv3_m{i}", m)
    dfl_weight = detect.dfl.state_dict()["conv.weight"].detach().numpy()
    # dfl_weight = dfl_weight.reshape((1, 1, dfl_weight.shape[1], 1))
    writer.add_tensor(prefix + "_dfl_weights", dfl_weight.astype(np.float16), raw_shape=dfl_weight.shape)

        

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: %s <yolov8.pt>" % sys.argv[0])
        sys.exit(1)

    outfile = "yolov8n.gguf"
    gguf_writer = gguf.GGUFWriter(outfile, "yolov8n")
    yolo = YOLO(sys.argv[1])
    
    # print(f"{yolo.model.model=}")
    # exit(0)
    model = yolo.model.model

    # torch_model: DetectionModel = model["model"]
    # print(f"{torch_model=}")
    # mods = list(list(torch_model.children())[0].children())

    # x = torch.rand(1, 3, 640, 640, dtype=torch.float)
    # print(f"input, {x.shape=}")
    # y = []
    # for i, mod in enumerate(model):
        
    #     if not isinstance(mod.f, int) and len(mod.f) > 1:
    #         print(f"{mod.f=}")
    #         x = mod([y[i] for i in mod.f])
    #     else:
    #         x = mod(x)

    #     y.append(x)
    #     if isinstance(x, torch.Tensor):
    #         print(f"{i=}, {x.shape=}")
    #     else:
    #         print(f"{i=}, {type(x)=}")

    # exit(0)
    for i, mod in enumerate(model):
        
        if isinstance(mod, Conv):
            print(f"m{i}: Conv")
            gguf_writer.add_string(f"m{i}_type", "conv")
            save_conv(gguf_writer, f"m{i}", mod)
        elif isinstance(mod, C2f):
            print(f"m{i}: C2f")
            gguf_writer.add_string(f"m{i}_type", "c2f")
            save_c2f(gguf_writer, f"m{i}", mod)
        elif isinstance(mod, SPPF):
            print(f"m{i}: SPPF")
            gguf_writer.add_string(f"m{i}_type", "sppf")
            save_sppf(gguf_writer, f"m{i}", mod)
        elif isinstance(mod, torch.nn.Upsample):
            print(f"m{i}: Upsample")
            gguf_writer.add_string(f"m{i}_type", "upsample")
            gguf_writer.add_string(f"m{i}_upsample_mode", mod.mode)
            
            gguf_writer.add_float32(f"m{i}_scale_factor", mod.scale_factor)
            print(f"m{i}_scale_factor: {mod.scale_factor}")
        elif isinstance(mod, Concat):
            print(f"m{i}: Concat")
            gguf_writer.add_string(f"m{i}_type", "concat")
            gguf_writer.add_int32(f"m{i}_dim", mod.d)
        elif isinstance(mod, Detect):
            print(f"m{i}: Detect")
            gguf_writer.add_string(f"m{i}_type", "detect")
            save_detect(gguf_writer, f"m{i}", mod)
        else:
            print(mod)
            print(f"m{i} is a {type(mod)}")
            raise NotImplementedError
    
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()