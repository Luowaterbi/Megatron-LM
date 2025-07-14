#!/bin/bash

SAVE_NAME=$1

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# 从 run_oc1.5.sh 迁移的路径和基本配置
ROOT="/step3-abla/siming"
MEGATRON_PATH="/step3-abla/lxz/oc/Megatron-LM"
# MEGATRON_PATH="./"
CHECKPOINT_PATH="/step3-abla/lxz/oc/save/checkpoint/${SAVE_NAME}/"
TENSORBOARD_LOGS_PATH="/step3-abla/lxz/oc/save/tensorboard/${SAVE_NAME}/"
PRETRAIN_CHECKPOINT_PATH="/step3-abla/lxz/oc/save/checkpoint/iter_0484865/"

# Create directories if they don't exist
mkdir -p "$CHECKPOINT_PATH"
mkdir -p "$TENSORBOARD_LOGS_PATH"

# Distributed training setup - 使用 stepmind 配置
GPUS_PER_NODE=${PROC_PER_NODE:-8}
NUM_NODES=${NODE_COUNT:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-9899}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script
PRETRAIN_SCRIPT_PATH="${MEGATRON_PATH}/pretrain_gpt.py"

# 从 run_oc1.5.sh 迁移的模型参数
NUM_LAYERS=24
HIDDEN_SIZE=2240
NUM_ATTN_HEADS=14
ROTARY_BASE=10000000
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1024
SEQ_LENGTH=8192
MAX_POSITION_EMBEDDINGS=8192
VOCAB_SIZE=96539
FFN_HIDDEN_SIZE=6144

# 从 run_oc1.5.sh 迁移的训练参数
LR=3e-4
MIN_LR=1e-5
TRAIN_TOKENS=1000000000000
LR_DECAY_TOKENS=1000000000000
WARMUP_TOKENS=$(( 500 * ${GLOBAL_BATCH_SIZE} * ${SEQ_LENGTH} ))

# 计算训练步数
TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LENGTH} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LENGTH} ))
LR_DECAY_ITERS=$(( ${LR_DECAY_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LENGTH} ))

# 从 run_oc1.5.sh 迁移的数据路径
DATASET_PATH="${ROOT}/pt_data/opc/opencoder_github_merge/oc-pt-c_0619 \
${ROOT}/pt_data/opc/opencoder_github_merge/oc-pt-code-others_0619 \
${ROOT}/pt_data/opc/opencoder_github_merge/oc-pt-cpp_0619 \
${ROOT}/pt_data/opc/opencoder_github_merge/oc-pt-csharp_0619 \
${ROOT}/pt_data/opc/opencoder_github_merge/oc-pt-data_0619 \
${ROOT}/pt_data/opc/opencoder_github_merge/oc-pt-go_0619 \
${ROOT}/pt_data/opc/opencoder_github_merge/oc-pt-html_0619 \
${ROOT}/pt_data/opc/opencoder_github_merge/oc-pt-java_0619 \
${ROOT}/pt_data/opc/opencoder_github_merge/oc-pt-javascript_0619 \
${ROOT}/pt_data/opc/opencoder_github_merge/oc-pt-php_0619 \
${ROOT}/pt_data/opc/opencoder_github_merge/oc-pt-python_0619 \
${ROOT}/pt_data/opc/opencoder_github_merge/oc-pt-text_0619 \
${ROOT}/pt_data/opc/opencoder_github_merge/oc-last-starcoder2_0619 \
${ROOT}/pt_data/opc/opencoder_github_merge/oc-last-jupyter_0619 \
${ROOT}/pt_data/opc/new_github_merge/new-pt-c_0619 \
${ROOT}/pt_data/opc/new_github_merge/new-pt-cpp_0619 \
${ROOT}/pt_data/opc/new_github_merge/new-pt-csharp_0619 \
${ROOT}/pt_data/opc/new_github_merge/new-pt-go_0619 \
${ROOT}/pt_data/opc/new_github_merge/new-pt-html_0619 \
${ROOT}/pt_data/opc/new_github_merge/new-pt-java_0619 \
${ROOT}/pt_data/opc/new_github_merge/new-pt-javascript_0619 \
${ROOT}/pt_data/opc/new_github_merge/new-pt-others_0619 \
${ROOT}/pt_data/opc/new_github_merge/new-pt-php_0619 \
${ROOT}/pt_data/opc/new_github_merge/new-pt-python_0619"

# Model parallelism (从 run_oc1.5.sh 迁移)
TP_SIZE=1
CP_SIZE=1
PP_SIZE=1

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# 模型架构参数 - 基于 run_oc1.5.sh 的配置
MODEL_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTN_HEADS
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base $ROTARY_BASE
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.02
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --normalization RMSNorm
    --norm-epsilon 1e-5
)

# 训练参数 - 基于 run_oc1.5.sh 的配置
TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS
    --lr-decay-iters $LR_DECAY_ITERS
    --lr-warmup-iters $LR_WARMUP_ITERS
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style constant
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --bf16
    --ckpt-format torch
)

# 模型并行参数
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
)

# 分布式优化器参数
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
)

# 数据参数 - 使用 NullTokenizer 如原脚本
DATA_ARGS_LIST=(
    --data-path "$DATASET_PATH"
    --tokenizer-type NullTokenizer
    --vocab-size $VOCAB_SIZE
    --split '100,0,0'
    --num-workers 8
)

# Flash Attention
FLASH_ARGS=(
    --use-flash-attn
)

# 评估和日志参数
EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --eval-interval 1000
    --eval-iters 50
    --save-interval 250
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --distributed-timeout-minutes 15
    --save "$CHECKPOINT_PATH"
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --seed 42
)

LOAD_CHECKPOINT_PATH=""
LOAD_TYPE=""
if [ -d "$CHECKPOINT_PATH" ] && [ "$(ls -A "$CHECKPOINT_PATH" 2>/dev/null)" ]; then
    echo "发现当前实验的续训检查点: $CHECKPOINT_PATH"
    LOAD_CHECKPOINT_PATH="$CHECKPOINT_PATH"
    LOAD_TYPE="continue"
elif [ -d "$PRETRAIN_CHECKPOINT_PATH" ]; then
    echo "发现预训练检查点: $PRETRAIN_CHECKPOINT_PATH"
    LOAD_CHECKPOINT_PATH="$PRETRAIN_CHECKPOINT_PATH"
    LOAD_TYPE="pretrain"
else
    echo "未找到可用检查点，从头开始训练"
fi

if [ -n "$LOAD_CHECKPOINT_PATH" ]; then
    EVAL_AND_LOGGING_ARGS+=(
        --load "$LOAD_CHECKPOINT_PATH"
    )
    
    if [ "$LOAD_TYPE" = "pretrain" ]; then
        EVAL_AND_LOGGING_ARGS+=(
            --finetune
            --no-load-optim
            --no-load-rng
        )
    fi
fi

cd "$MEGATRON_PATH"

MTP_ARGS=(
    --mtp-num-layers 2                   # MTP 层数
    --mtp-loss-scaling-factor 0.1         # MTP 损失缩放因子
)


echo "========================================"
echo "开始训练，配置如下："
echo "- 实验名称: $SAVE_NAME"
echo "- Megatron 路径: $MEGATRON_PATH"
echo "- 检查点保存路径: $CHECKPOINT_PATH"
echo "- TensorBoard 路径: $TENSORBOARD_LOGS_PATH"
echo "- 加载检查点: $LOAD_CHECKPOINT_PATH"
echo "- 模型层数: $NUM_LAYERS"
echo "- 隐藏层大小: $HIDDEN_SIZE"
echo "- 注意力头数: $NUM_ATTN_HEADS"
echo "- 序列长度: $SEQ_LENGTH"
echo "- 全局批次大小: $GLOBAL_BATCH_SIZE"
echo "- 学习率: $LR"
echo "- 训练步数: $TRAIN_ITERS"
echo "========================================"

torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DDP_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${FLASH_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${MTP_ARGS[@]}

set +x