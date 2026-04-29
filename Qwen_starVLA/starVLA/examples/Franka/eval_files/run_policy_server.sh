#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} 
export star_vla_python=python

# 🌟 关键修复：去掉了末尾的反斜杠
your_ckpt=/mnt/bigdata/minghua/starVLA_project/weights/starvla_qwen_model/checkpoints/steps_40000_pytorch_model.pt

gpu_id=0
port=5694

################# star Policy Server ######################
echo "🚀 准备在 5694 端口启动 starVLA 服务端..."

CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16