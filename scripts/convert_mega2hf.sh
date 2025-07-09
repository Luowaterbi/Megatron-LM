#!/bin/bash

# Megatron 分布式检查点转 HuggingFace 格式

# 配置参数
MEGATRON_CHECKPOINT_DIR="/step3-abla/lxz/oc/save/code-1p5b-ct-dist-v1/checkpoint/iter_0001000"
TEMP_UNIFIED_DIR="/tmp/unified_checkpoint"
LEGACY_OUTPUT_DIR="/step3-abla/lxz/oc/save/code-1p5b-ct-dist-v1/checkpoint/iter_0001000_legacy"  # 修改为您的输出路径

# 创建临时和输出目录
mkdir -p "$TEMP_UNIFIED_DIR"
mkdir -p "$LEGACY_OUTPUT_DIR"

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:/home/i-luoxianzhen/oc1.5/Megatron-LM"

# 进入 Megatron-LM 目录
cd /home/i-luoxianzhen/oc1.5/Megatron-LM

# 第一步：将分布式检查点转换为统一格式
echo "第一步：将分布式检查点转换为统一格式..."
python3 tools/checkpoint/convert.py \
    --model-type GPT \
    --loader core \
    --saver legacy \
    --load-dir "$MEGATRON_CHECKPOINT_DIR" \
    --save-dir "$LEGACY_OUTPUT_DIR" \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1

# # 第二步：从统一格式转换为 HuggingFace 格式
# echo "第二步：从统一格式转换为 HuggingFace 格式..."
# python tools/checkpoint/convert.py \
#     --model-type GPT \
#     --loader legacy \
#     --saver llama_mistral \
#     --load-dir "$TEMP_UNIFIED_DIR" \
#     --save-dir "$HF_OUTPUT_DIR" \
#     --model-size llama3 \
#     --checkpoint-type hf \
#     --bf16

# # 清理临时文件
# echo "清理临时文件..."
# rm -rf "$TEMP_UNIFIED_DIR"

# echo "转换完成！HuggingFace 格式的模型已保存到: $HF_OUTPUT_DIR"