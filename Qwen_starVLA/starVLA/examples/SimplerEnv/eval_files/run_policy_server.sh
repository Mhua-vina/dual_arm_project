#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} 
export star_vla_python=python

# 🌟 指向你刚才搜出来的 Qwen3-VL 文件夹（不是文件！）
your_ckpt=/mnt/bigdata/minghua/starVLA_project/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct

gpu_id=0
port=5694

################# star Policy Server ######################
echo "🚀 准备在 5694 端口启动 starVLA 服务端..."

CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16