#!/bin/bash

# 设置 CUDA 可见设备（如果需要的话）
export CUDA_VISIBLE_DEVICES=4

# BASE_PATH=/data/wdy/Downloads/models/Qwen
# MODEL_NAME=Qwen2.5-VL-3B-Instruct
BASE_PATH=/data/wdy/StarVector/star-vector/sft/LLaMA-Factory/output
MODEL_NAME=qwen2_5vl_lora_sft_20k
MODEL_PATH=$BASE_PATH/$MODEL_NAME

# 启动 vLLM 服务
vllm serve $MODEL_PATH \
    --host 0.0.0.0 \
    --port 8011 \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --served-model-name $MODEL_NAME
