#!/bin/bash

# 设置路径
HF_CHECKPOINT_PATH="/step3-abla/siming/ckpts/hf/code_expV59/iter_0484865"
MEGATRON_CHECKPOINT_PATH="/step3-abla/lxz/oc/save/checkpoint/iter_0484865_vocab_96539"

# 创建输出目录
mkdir -p "$MEGATRON_CHECKPOINT_PATH"

# 转换脚本路径
CONVERT_SCRIPT="tools/checkpoint/convert.py"

# 从你的训练配置中获取模型参数
NUM_LAYERS=24
HIDDEN_SIZE=2240
NUM_ATTN_HEADS=14
VOCAB_SIZE=96539
SEQ_LENGTH=4096
FFN_HIDDEN_SIZE=6144

# 模型并行设置（与你的训练配置保持一致）
TP_SIZE=1
PP_SIZE=1

cd "$MEGATRON_PATH"

echo "开始转换 HuggingFace 检查点到 Megatron 格式..."
echo "源路径: $HF_CHECKPOINT_PATH"
echo "目标路径: $MEGATRON_CHECKPOINT_PATH"

python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader llama_mistral  \
    --saver core \
    --target-tensor-parallel-size $TP_SIZE \
    --target-pipeline-parallel-size $PP_SIZE \
    --load-dir "$HF_CHECKPOINT_PATH" \
    --save-dir "$MEGATRON_CHECKPOINT_PATH" \
    --tokenizer-model "$HF_CHECKPOINT_PATH" \
    --model-size llama3 \
    --checkpoint-type hf \
    --bf16

echo "转换完成！"
echo "转换后的检查点位置: $MEGATRON_CHECKPOINT_PATH"

# 验证转换结果
echo "验证转换结果..."
ls -la "$MEGATRON_CHECKPOINT_PATH"