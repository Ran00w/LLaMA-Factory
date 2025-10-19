#!/usr/bin/env bash
set -euo pipefail

# 批量遍历一段学习率区间并调用 train.sh 训练，同时生成对比报告。

usage() {
  cat <<'USAGE'
用法:
  scripts/sweep_learning_rate.sh --start <min_lr> --end <max_lr> --step <step>
  scripts/sweep_learning_rate.sh --values "1e-4 5e-5 1e-5"

其它常用参数:
  --train-script <path>   指定训练脚本，默认 ./train.sh
  --run-prefix <prefix>   运行名前缀（会追加 lr 和时间戳）
  --output-root <dir>     用于推导输出目录，默认为 train.sh 中的默认值
  --report-dir <dir>      报告输出位置，默认 reports/
  --dry-run               仅打印计划，不实际执行训练
  -h, --help              查看帮助

说明:
  该脚本会针对每个学习率设置环境变量 LEARNING_RATE 和 RUN_NAME 后执行 train.sh。
  每次运行结束后，会尝试从 trainer_state.json 中读取 best_metric 与最后一次 loss，
  并将结果写入 CSV 报告，方便对比。
USAGE
}

START_LR=""
END_LR=""
STEP_LR=""
VALUES=()
TRAIN_SCRIPT="./train.sh"
RUN_PREFIX_OVERRIDE=""
OUTPUT_ROOT_OVERRIDE=""
REPORT_DIR="reports"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start)
      START_LR=$2
      shift 2
      ;;
    --end)
      END_LR=$2
      shift 2
      ;;
    --step)
      STEP_LR=$2
      shift 2
      ;;
    --values)
      IFS=' ' read -r -a VALUES <<<"$2"
      shift 2
      ;;
    --train-script)
      TRAIN_SCRIPT=$2
      shift 2
      ;;
    --run-prefix)
      RUN_PREFIX_OVERRIDE=$2
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT_OVERRIDE=$2
      shift 2
      ;;
    --report-dir)
      REPORT_DIR=$2
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac

done

if [[ ${#VALUES[@]} -eq 0 ]]; then
  if [[ -z "${START_LR}" || -z "${END_LR}" || -z "${STEP_LR}" ]]; then
    echo "错误: 需要提供 --start/--end/--step，或通过 --values 指定学习率列表。" >&2
    usage
    exit 1
  fi

  mapfile -t VALUES < <(
    python - <<'PY' "${START_LR}" "${STEP_LR}" "${END_LR}"
import sys
from decimal import Decimal, getcontext

getcontext().prec = 20
start = Decimal(sys.argv[1])
step = Decimal(sys.argv[2])
end = Decimal(sys.argv[3])

if step == 0:
    raise SystemExit("步长 step 不能为 0")

direction = 1 if step > 0 else -1
if (end - start) * step < 0:
    raise SystemExit("step 方向与区间不一致，请检查 start/end/step")

values = []
current = start
idx = 0
while True:
    values.append(current.normalize())
    current = start + (idx + 1) * step
    idx += 1
    if direction > 0:
        if current > end + Decimal("1e-18"):
            break
    else:
        if current < end - Decimal("1e-18"):
            break
print("\n".join(str(v) for v in values))
PY
  )
fi

TOTAL=${#VALUES[@]}
if [[ ${TOTAL} -eq 0 ]]; then
  echo "未生成任何学习率。请检查参数。" >&2
  exit 1
fi

if [[ ! -x "${TRAIN_SCRIPT}" ]]; then
  echo "训练脚本 ${TRAIN_SCRIPT} 不存在或不可执行。" >&2
  exit 1
fi

# 与 train.sh 中默认值保持一致，可通过环境变量或参数覆盖
DEFAULT_OUTPUT_ROOT="/media/dpctc/4TB1/lr/ner-lora"
DEFAULT_RUN_PREFIX="ner-sft-lora"

RUN_PREFIX_EFFECTIVE=${RUN_PREFIX_OVERRIDE:-${RUN_PREFIX:-${DEFAULT_RUN_PREFIX}}}
OUTPUT_ROOT_EFFECTIVE=${OUTPUT_ROOT_OVERRIDE:-${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "${REPORT_DIR}"
REPORT_FILE="${REPORT_DIR}/lr_sweep_${TIMESTAMP}.csv"

echo "learning_rate,run_name,output_dir,status,best_metric,last_train_loss,log_file" > "${REPORT_FILE}"

echo "[信息] 将执行 ${TOTAL} 次训练，输出报告：${REPORT_FILE}"
${DRY_RUN} && echo "[信息] Dry-run 模式，训练不会实际执行。"

idx=0
for lr in "${VALUES[@]}"; do
  idx=$((idx + 1))
  lr_str=$(printf "%s" "${lr}")
  lr_slug=$(printf "%s" "${lr_str}" | tr '[:upper:]' '[:lower:]' | sed 's/[^0-9a-z]/-/g; s/-\{2,\}/-/g; s/^-//; s/-$//')
  if [[ -z "${lr_slug}" ]]; then
    lr_slug="lr"
  fi
  run_name="${RUN_PREFIX_EFFECTIVE}-lr${lr_slug}-${TIMESTAMP}-${idx}"
  output_dir="${OUTPUT_ROOT_EFFECTIVE}/${run_name}"

  echo
  echo "[信息] (${idx}/${TOTAL}) 学习率：${lr_str}"
  echo "[信息] 运行名称：${run_name}"
  echo "[信息] 输出目录：${output_dir}"

  status="success"
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[信息] Dry-run 跳过实际训练。"
  else
    if ! OUTPUT_ROOT="${OUTPUT_ROOT_EFFECTIVE}" RUN_PREFIX="${RUN_PREFIX_EFFECTIVE}" RUN_NAME="${run_name}" LEARNING_RATE="${lr_str}" "${TRAIN_SCRIPT}"; then
      status="failed"
      echo "[警告] 训练失败，结果将标记为 failed。"
    fi
  fi

  log_file=""
  best_metric="N/A"
  last_train_loss="N/A"

  if [[ "${status}" == "success" ]]; then
    if [[ -d "${output_dir}" ]]; then
      if log_path=$(ls -t "${output_dir}"/train_*.log 2>/dev/null | head -n1); then
        if command -v realpath >/dev/null 2>&1; then
          log_file=$(realpath "${log_path}")
        else
          log_file=$(readlink -f "${log_path}" 2>/dev/null || echo "${log_path}")
        fi
      fi
      state_file="${output_dir}/trainer_state.json"
      if [[ -f "${state_file}" ]]; then
        readarray -t metrics < <(
          python - <<'PY' "${state_file}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit
try:
    data = json.loads(path.read_text())
except json.JSONDecodeError:
    raise SystemExit
best_metric = data.get("best_metric", "N/A")
last_loss = "N/A"
for record in reversed(data.get("log_history", [])):
    if "loss" in record:
        last_loss = record["loss"]
        break
print(best_metric)
print(last_loss)
PY
        )
        if [[ ${#metrics[@]} -ge 1 ]]; then
          best_metric=${metrics[0]}
        fi
        if [[ ${#metrics[@]} -ge 2 ]]; then
          last_train_loss=${metrics[1]}
        fi
      fi
    else
      status="failed"
      echo "[警告] 未找到输出目录 ${output_dir}，标记为 failed。"
    fi
  fi

  printf '"%s","%s","%s","%s","%s","%s","%s"\n' \
    "${lr_str}" \
    "${run_name}" \
    "${output_dir}" \
    "${status}" \
    "${best_metric}" \
    "${last_train_loss}" \
    "${log_file}" >> "${REPORT_FILE}"
done

printf '\n[信息] 学习率扫描完成。报告路径：%s\n' "${REPORT_FILE}"

python - <<'PY' "${REPORT_FILE}"
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open(newline="")))
if not rows:
    print("[提示] 报告为空。")
    raise SystemExit
headers = list(rows[0].keys())
widths = {h: len(h) for h in headers}
for row in rows:
    for key in headers:
        widths[key] = max(widths[key], len(str(row[key])))
print(" | ".join(f"{h:<{widths[h]}}" for h in headers))
print("-+-".join("-" * widths[h] for h in headers))
for row in rows:
    print(" | ".join(f"{str(row[h]):<{widths[h]}}" for h in headers))
PY


# ./scripts/sweep_learning_rate.sh --start 5e-5 --end 2e-4 --step 1e-5