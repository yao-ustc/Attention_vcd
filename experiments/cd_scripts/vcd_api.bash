#!/bin/bash

# 参数传递
seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"adversarial"}
model_path=${4:-"/home/yaozhiyuan/Model/liuhaotian/llava-v1.5-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.2}
noise_step=${7:-300}

# hook 参数
# 修改为只传递参数名
use_hl="--use-hl"

# 选择的层数，范围从 0 到 31
layer_index=31

# 是否使用 hook
#modify_attention="None"
modify_attention="--modify-attention"

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
mod_ratio=0.5

# 该参数仅在 hook-mode 为 "add_noise" 时生效，表示在注意力分数上添加的噪声的标准差。
noise_std=0.1

# 图像路径选择
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/home/yaozhiyuan/Data/pope/val2014/coco
else
  image_folder=./data/gqa/images
fi

# 获取动态生成的 answers-file 文件名
answers_file="./output/layer${layer_index}_"

if [[ $modify_attention == "--modify-attention" ]]; then
  answers_file+="启用vcd_负样本使用全黑图像" 
  #answers_file+="modify_${hook_mode}_ratio${mod_ratio}_noise${noise_std}"
else
  answers_file+="no_modify_attention"
fi

answers_file+=".jsonl"

# 启动 Python 脚本
python ./eval/object_hallucination_vqa_llava.py \
  --model-path ${model_path} \
  --question-file /home/yaozhiyuan/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
  --image-folder ${image_folder} \
  --answers-file $answers_file \
  --use_cd \
  --cd_alpha $cd_alpha \
  --cd_beta $cd_beta \
  --noise_step $noise_step \
  --seed ${seed} \
  #$use_hl \
  --layer-index $layer_index \
  #$modify_attention \
  --hook-mode $hook_mode \
  --mod-ratio $mod_ratio \
  --noise-std $noise_std
