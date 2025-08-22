#!/bin/bash

# =============================================================================
# 分层化鱼嘴行为分析 - Slurm 作业脚本 (支持文件夹或单文件输入)
# =============================================================================

# --- Slurm 配置区 ---
#SBATCH --job-name=hierarchical_fish_analysis
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com # !!! 请替换为您的邮箱地址 !!!

# --- 用户配置区 ---
# !!! 请根据您的实际环境和需求修改以下配置 !!!

# 1. 项目根目录
PROJECT_DIR="/gpfs/work/bio/shuleihe23/fish_mouth/fish_mouth"

# 2. 要调用的 Python 脚本名称
PYTHON_SCRIPT="pipeline.py"

# 3. 输入路径 (可以是单个视频文件或包含视频的文件夹)
INPUT_PATH="/gpfs/work/bio/shuleihe23/data/2025.8.1_Habituation"
# --- 或者单个文件示例: ---
# INPUT_PATH="/gpfs/work/bio/shuleihe23/fish_mouth/fish_mouth/script_for_5min_analysis/MVI_7905.mp4"

# 4. 输出结果的总目录
OUTPUT_DIR="/gpfs/work/bio/shuleihe23/fish_mouth/fish_mouth/results_hierarchical/8.13"

# 5. 模型文件路径
HEAD_MODEL_PATH="/gpfs/work/bio/shuleihe23/fish_mouth/model/head_detect_v2.pt"
EYE_MODEL_PATH="/gpfs/work/bio/shuleihe23/fish_mouth/model/fisheyes.pt"
MOUTH_MODEL_PATH="/gpfs/work/bio/shuleihe23/fish_mouth/model/fishmouth_R3.1.pt"

# 6. Conda 环境名称
CONDA_ENV="yolov12"

# --- 环境准备与验证区 ---
echo "========================================================"
echo "🐟 分层化鱼嘴行为分析任务开始"
echo "========================================================"
echo "作业ID: $SLURM_JOB_ID | 开始时间: $(date)"
echo "========================================================"

# ... (此处省略了与之前版本相同的环境准备代码) ...
source ~/.bashrc
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR" || exit 1

# --- 路径验证区 (核心修改) ---
echo "🤖 验证所有文件和目录路径..."
PYTHON_INPUT_ARGS=""
if [ -d "$INPUT_PATH" ]; then
    echo "  ✅ 输入路径是一个目录: $INPUT_PATH"
    PYTHON_INPUT_ARGS="--input_dir \"$INPUT_PATH\""
elif [ -f "$INPUT_PATH" ]; then
    echo "  ✅ 输入路径是一个文件: $INPUT_PATH"
    PYTHON_INPUT_ARGS="--input_file \"$INPUT_PATH\""
else
    echo "❌ 致命错误：输入路径不存在或类型未知: $INPUT_PATH" >&2
    exit 1
fi

paths_to_check=(
    "$HEAD_MODEL_PATH"
    "$EYE_MODEL_PATH"
    "$MOUTH_MODEL_PATH"
    "$PYTHON_SCRIPT"
)
for path in "${paths_to_check[@]}"; do
    if [ ! -e "$path" ]; then
        echo "❌ 致命错误：路径不存在: $path" >&2
        exit 1
    fi
done
echo "  ✅ 所有路径验证通过。"


# --- 任务执行区 ---
echo ""
echo "========================================================"
echo "🚀 开始执行分层分析流水线..."
echo "========================================================"
echo "Python 脚本: $PYTHON_SCRIPT"
echo "输入路径: $INPUT_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "========================================================"

START_TIME=$(date +%s)

# 使用 eval 来正确处理带引号的参数
eval python -u "$PYTHON_SCRIPT" \
    $PYTHON_INPUT_ARGS \
    --output_dir "\"$OUTPUT_DIR\"" \
    --head_model "\"$HEAD_MODEL_PATH\"" \
    --eye_model "\"$EYE_MODEL_PATH\"" \
    --mouth_model "\"$MOUTH_MODEL_PATH\"" \
    2>&1 | tee -a "$PROJECT_DIR/logs/analysis_${SLURM_JOB_ID}.log"

ANALYSIS_EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)


# 开启调试模式，打印所有执行的命令
set -x

# 执行主要的串行分析脚本
python -u "$PYTHON_SCRIPT" \
    --input_dir "$INPUT_VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --head_model "$HEAD_MODEL_PATH" \
    --eye_model "$EYE_MODEL_PATH" \
    --mouth_model "$MOUTH_MODEL_PATH" \
    2>&1 | tee -a "$PROJECT_DIR/logs/analysis_${SLURM_JOB_ID}.log"

# 检查Python脚本的退出码
ANALYSIS_EXIT_CODE=${PIPESTATUS[0]}

# 关闭调试模式
set +x

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# --- 结果统计与清理区 ---
echo ""
echo "========================================================"
if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo "🎉 任务执行成功完成！ [$(date)]"
else
    echo "❌ 任务执行失败，退出码: $ANALYSIS_EXIT_CODE [$(date)]"
fi
echo "========================================================"

echo "📊 任务统计信息:"
echo "  - 总用时: $((DURATION / 3600))小时 $((DURATION % 3600 / 60))分钟 $((DURATION % 60))秒"

# 统计输出结果
if [ -d "$OUTPUT_DIR" ]; then
    SUMMARY_FILE="$OUTPUT_DIR/batch_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "  - 汇总报告: ✅ 已生成 ($SUMMARY_FILE)"
    else
        echo "  - 汇总报告: ❌ 未找到"
    fi
else
    echo "  - 输出目录: ❌ 未创建"
fi

echo ""
echo "========================================================"
echo "🎯 任务完成于: $(date)"
echo "========================================================"

exit $ANALYSIS_EXIT_CODE
