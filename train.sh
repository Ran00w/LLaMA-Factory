#!/usr/bin/env bash
set -euo pipefail

# 运行脚本前可通过环境变量覆盖默认配置。
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/data2/liran/model/qwen2.5-7B-Instruct}
DATASET_NAME=${DATASET_NAME:-ner_sft_dataset}
DATASET_DIR=${DATASET_DIR:-data}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data2/liran/model/ner-lora}
RUN_PREFIX=${RUN_PREFIX:-ner-sft-lora}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET=${LORA_TARGET:-all}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-qwen}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
PREPROCESSING_WORKERS=${PREPROCESSING_WORKERS:-8}
DATALOADER_WORKERS=${DATALOADER_WORKERS:-4}
CUTOFF_LEN=${CUTOFF_LEN:-2048}
SEED=${SEED:-42}
BF16=${BF16:-true}
TF32=${TF32:-false}
PLOT_LOSS=${PLOT_LOSS:-true}
REPORT_TO=${REPORT_TO:-none}
DDP_TIMEOUT=${DDP_TIMEOUT:-180000000}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME=${RUN_NAME:-"${RUN_PREFIX}-${TIMESTAMP}"}
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_FILE="${OUTPUT_DIR}/train_${TIMESTAMP}.log"

mkdir -p "${OUTPUT_DIR}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

finalize() {
  local status=$1
  local end_time
  end_time=$(date +"%F %T")
  if [[ ${status} -eq 0 ]]; then
    echo "[信息] 训练已于 ${end_time} 完成"
  else
    echo "[错误] 训练于 ${end_time} 失败（退出码：${status}）"
  fi
}
trap 'status=$?; set +e; finalize ${status}' EXIT

START_TIME=$(date +"%F %T")
echo "[信息] 训练开始时间：${START_TIME}"
echo "[信息] 运行名称：${RUN_NAME}"
echo "[信息] 输出目录：${OUTPUT_DIR}"
echo "[信息] 基础模型：${MODEL_NAME_OR_PATH}"
echo "[信息] 数据集：${DATASET_NAME}（目录：${DATASET_DIR}）"

if [[ ! -f "${DATASET_DIR}/ner_sft.json" ]]; then
  echo "[警告] 预期的数据集文件 ${DATASET_DIR}/ner_sft.json 未找到，添加该文件前训练会失败。"
fi

export WANDB_DISABLED=${WANDB_DISABLED:-true}
export CUDA_VISIBLE_DEVICES=1

# 定义要执行的命令和参数
CMD=(
  llamafactory-cli train
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --trust_remote_code true
  --stage sft
  --do_train true
  --finetuning_type lora
  --lora_rank "${LORA_RANK}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --lora_target "${LORA_TARGET}"
  --dataset "${DATASET_NAME}"
  --template "${PROMPT_TEMPLATE}"
  --overwrite_cache true
  --preprocessing_num_workers "${PREPROCESSING_WORKERS}"
  --dataloader_num_workers "${DATALOADER_WORKERS}"
  --output_dir "${OUTPUT_DIR}"
  --run_name "${RUN_NAME}"
  --report_to "${REPORT_TO}"
  --logging_steps "${LOGGING_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --overwrite_output_dir true
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
  --warmup_ratio "${WARMUP_RATIO}"
  --seed "${SEED}"
  --bf16 "${BF16}"
  --gradient_checkpointing true
)

if [[ ${PLOT_LOSS} == true ]]; then
  CMD+=(--plot_loss true)
fi

echo "[信息] 启动命令：${CMD[*]}"
"${CMD[@]}"
