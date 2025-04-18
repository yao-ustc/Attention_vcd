import argparse
import torch
import numpy as np

from llava_api.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava_api.conversation import conv_templates, SeparatorStyle
from llava_api.model.builder import load_pretrained_model
from llava_api.utils import disable_torch_init
from llava_api.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava_api.transformers.generation.stopping_criteria import MaxNewTokensCriteria
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
import re

def save_attention_to_txt(attn_output, filename):
    # attn_output 是一个 2D 或 3D tensor（比如 batch_size x num_heads x seq_len），需要reshape成 2D 写入
    with open(filename, 'a') as f:  # 使用 'a' 模式来追加内容，而不是覆盖
        for i in range(attn_output.shape[0]):  # 遍历 batch
            f.write(f"Batch {i}:\n")
            for j in range(attn_output.shape[1]):  # 遍历头部
                f.write(f"  Head {j}:\n")
                for k in range(attn_output.shape[2]):  # 遍历序列长度
                    f.write(f"    {attn_output[i, j, k].tolist()}\n")  # 写入注意力分数
            f.write("\n")


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def getmask(args):
    # Model
    disable_torch_init()

    tokenizer, model, image_processor, context_len = args.tokenizer, args.model, args.image_processor, args.context_len
    hl = args.h2
    hl.reinit()
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in args.model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in args.model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in args.model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if isinstance(args.image_file, str):
        image_files = image_parser(args)
        images = load_images(image_files)
    elif isinstance(args.image_file, Image.Image):
        images = [args.image_file]
    else:
        raise ValueError("image_file should be str or PIL.Image")
    
    images = [image.convert('RGB') if image.mode != 'RGB' else image for image in images]
        
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = [
        KeywordsStoppingCriteria(keywords, tokenizer, input_ids),
        MaxNewTokensCriteria(input_ids.shape[1], args.max_new_tokens)
    ]


    with torch.inference_mode():
        model.eval()        
        output_ids = model.generate(
            input_ids,
            images=None,
            images_cd=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=stopping_criteria,

        )
        #hl.modify_attentions()
        #print("输出的 token ids:", output_ids[0].tolist())
        #print(f"最小 token ID: {output_ids.min()}, 最大 token ID: {output_ids.max()}")
        #print(f"Tokenizer 词表大小: {tokenizer.vocab_size}")
        valid_output_ids = [tid for tid in output_ids[0].tolist() if tid >= 0]
        output_text = tokenizer.decode(valid_output_ids, skip_special_tokens=True)
        print("模型回答:", output_text)
    attention_output = hl.finalize().view(24,24)

    attention_output_h2 = hl.finalize()  # 获取 h2 记录的注意力
    #save_attention_to_txt(attention_output_h2, "attention_h2.txt")
    print("注意力分数已保存到文件.")
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )

    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:].cpu(), skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    return attention_output.detach(), outputs

def get_model(model_path = "llava-v1.5-7b"):
    model_path = model_path
    model_base = None
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
    )
    return tokenizer, model, image_processor, context_len, model_name
