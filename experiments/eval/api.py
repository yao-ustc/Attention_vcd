## Imports
import os, time, argparse, base64, requests, os, json, sys, datetime
from itertools import product
import warnings
import cv2
warnings.filterwarnings("ignore")
import sys

# 设置 sys.path，用于导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import cv2
from PIL import Image

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
import torchvision.transforms as T

from functions import getmask, get_model
from hook import hook_logger
#from DatasetManager.dataloader import get_data
from vcd_utils.vcd_sample import evolve_vcd_sampling

# 执行采样前初始化
evolve_vcd_sampling()
def readImg(p):
    return Image.open(p)

def toImg(t):
    return T.ToPILImage()(t)

def invtrans(mask, image, method = Image.BICUBIC):
    return mask.resize(image.size, method)

def normalize(mat, method = "max"):
    if method == "max":
        return (mat.max() - mat) / (mat.max() - mat.min())
    elif method == "min":
        return (mat - mat.min()) / (mat.max() - mat.min())
    else:
        raise NotImplementedError

def enhance(mat, coe=10):
    mat = mat - mat.mean()
    mat = mat / mat.std()
    mat = mat * coe
    mat = torch.sigmoid(mat)
    mat = mat.clamp(0,1)
    return mat

def revise_mask(patch_mask, kernel_size = 3, enhance_coe = 10):

    patch_mask = normalize(patch_mask, "min")
    patch_mask = enhance(patch_mask, coe = enhance_coe)

    assert kernel_size % 2 == 1
    padding_size = int((kernel_size - 1) / 2)
    conv = torch.nn.Conv2d(1,1,kernel_size = kernel_size, padding = padding_size, padding_mode = "replicate", stride = 1, bias = False)
    conv.weight.data = torch.ones_like(conv.weight.data) / kernel_size**2
    conv.to(patch_mask.device)

    patch_mask = conv(patch_mask.unsqueeze(0))[0]

    mask = patch_mask

    return mask

def blend_mask(image_path_or_pil_image, mask, key, enhance_coe, kernel_size, interpolate_method, folder):
    mask = revise_mask(mask.float(), kernel_size = kernel_size, enhance_coe = enhance_coe)
    mask = mask.detach().cpu()
    mask = toImg(mask.reshape(1,24,24))

    if isinstance(image_path_or_pil_image, str):
        image = readImg(image_path_or_pil_image)
    elif isinstance(image_path_or_pil_image, Image.Image):
        image = image_path_or_pil_image
    else:
        raise NotImplementedError

    mask = invtrans(mask, image, method = interpolate_method)
    # 将 mask 转换为 numpy 数组
    mask_np = np.array(mask).astype(np.uint8)
    # 应用 cv2.COLORMAP_JET 生成热力图
    heatmap = cv2.applyColorMap(mask_np, cv2.COLORMAP_JET)
    # 将 PIL 图像转换为 numpy 数组
    image_np = np.array(image).astype(np.uint8)
    # 调整 heatmap 的通道顺序以匹配 image_np
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # 融合热力图和原始图像
    alpha = 0.5  # 透明度，可根据需要调整
    merged_np = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    # 将 numpy 数组转换为 PIL 图像
    merged_image = Image.fromarray(merged_np)

    file_name = os.path.join(folder, f"{key}.jpg")
    merged_image.save(file_name)
    print(file_name)
    return merged_image

if __name__ == "__main__":
    # 直接指定参数
    image_file = "/home/yaozhiyuan/Data/pope/val2014/coco/COCO_val2014_000000000395.jpg"
    prompt = "Where and when was this picture taken?"
    print(f"image_file: {image_file}")
    print(f"prompt: {prompt}")
    model_path = "/home/yaozhiyuan/Model/liuhaotian/llava-v1.5-7b"

    # 其他参数设置为默认值
    #提取第i+1层的注意力
    layer_index = 20
    output_folder = "../../experiments"
    interpolate_method_name = "LANCZOS"
    # 用于增强 mask 的系数
    #将归一化后的掩码乘以 enhance_coe 系数，然后通过 sigmoid 函数进行非线性变换
    enhance_coe = 5
    #在 revise_mask 函数中对注意力掩码进行卷积操作
    kernel_size = 3

    exp_folder = f"{output_folder}/APILLaVA_mmvet_7b_{layer_index}"

    # 加载模型
    tokenizer, model, image_processor, context_len, model_name = get_model(model_path)

    h2 = hook_logger(model, model.device, layer_index=layer_index, modify_attention=True, name="h2")

    # 这里不再使用 get_data 函数，因为我们直接指定了图像
    # dataset = get_data(args.dataset)

    interpolate_method = getattr(Image, interpolate_method_name)
    mask_image_folder = os.path.join(exp_folder, f"{enhance_coe}_{kernel_size}_{interpolate_method_name}")
    os.makedirs(mask_image_folder, exist_ok=True)

    key = os.path.basename(image_file).split('.')[0]  # 简单使用文件名作为 key

    with torch.no_grad():
        mask_args = type('Args', (), {
            "h2": h2,  # h2 钩子
            "model_name": model_name,
            "model": model,
            "tokenizer": tokenizer,
            "image_processor": image_processor,
            "context_len": context_len,
            "query": prompt,
            "conv_mode": None,
            "image_file": image_file,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512,
        })()

        mask, output = getmask(mask_args)
        # 修改注意力分数
        #hl.modify_attentions()

        # 再次运行推理，使用修改后的注意力分数
        #mask, output = getmask(mask_args)
        merged_image = blend_mask(image_file, mask, key, enhance_coe, kernel_size, interpolate_method, mask_image_folder)