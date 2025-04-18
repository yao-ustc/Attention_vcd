#!/bin/bash

# 参数传递
seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"random"}
model_path=${4:-"/home/yaozhiyuan/Model/llava-v1.5-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.2}
noise_step=${7:-500}

# hook 参数
use_hl=True

# 选择的层数，范围从 0 到 31
layer_index=0

# 是否使用 hook
modify_attention=True

# hook 的修改方式，指定如何修改注意力
# 选项包括：
#制造负样本：
#   - "suppress_topk"：删除前 x%（mod-ratio）注意力最高的 token
#   - "add_noise"：在注意力值上添加噪声
#   - "shuffle"：随机打乱部分 token 的注意力值
#   - "zeroout"：随机清空部分 token 的注意力值
#制造正样本：
#   - "suppress_low"：删除前 x%（mod-ratio）注意力最低的 token
hook_mode="suppress_low"   

# 在进行注意力修改时的比例（例如 0.5 表示删除或修改前 50% 的 token）
# 例如，当 hook_mode 为 "suppress_topk" 时，将删除前 50% 的注意力值
# 当 hook_mode 为 "add_noise" 时，将给前 50% 的 token 添加噪声
mod_ratio=0.5

#该参数仅在 hook-mode 为 "add_noise" 时生效，表示在注意力分数上添加的噪声的标准差。较大的标准差会使得噪声的干扰更强，影响模型的推理。
#该参数允许调节噪声的强度，以实现不同强度的扰动效果。
noise_std=0.1

# 图像路径选择
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/home/yaozhiyuan/Data/pope/val2014/coco
else
  image_folder=./data/gqa/images
fi

# 获取动态生成的 answers-file 文件名
answers_file="./output/pope_llava7b_layer${layer_index}_"

if [[ $modify_attention == "True" ]]; then
  answers_file+="modify_${hook_mode}_ratio${mod_ratio}_noise${noise_std}"
else
  answers_file+="no_modify_attention"
fi

answers_file+=".jsonl"



python /home/yaozhiyuan/LLaVA/llava/eval/eval_pope.py \
    --annotation-dir /home/yaozhiyuan/LLaVA/playground/data/eval/pope/coco \
    --question-file /home/yaozhiyuan/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file "/home/yaozhiyuan/VCD-master/experiments/output/pope_llava7b_layer31_启用vcd_负样本采用修改注意力层_suppress_topk_ratio0.8_noise0.1.jsonl"
