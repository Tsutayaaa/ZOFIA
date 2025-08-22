#!/bin/bash

#SBATCH --job-name=direct_fish_analysis
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
#SBATCH --mail-user=Gan.Wang23@student.xjtlu.edu.cn

PROJECT_DIR="/gpfs/work/bio/shuleihe23/fish_mouth/fish_mouth"
PYTHON_SCRIPT="pipeline2.py"
INPUT_PATH="/gpfs/work/bio/shuleihe23/data/2025.7.29_GBR/MVI_7740.MP4"
OUTPUT_DIR="/gpfs/work/bio/shuleihe23/fish_mouth/results_direct2"
HEAD_MODEL_PATH="/gpfs/work/bio/shuleihe23/fish_mouth/model/head_detect.pt"
EYE_MODEL_PATH="/gpfs/work/bio/shuleihe23/fish_mouth/model/fisheyes3.0.pt"
MOUTH_MODEL_PATH="/gpfs/work/bio/shuleihe23/fish_mouth/model/fishmouth_R3.1.pt"
CONDA_ENV="yolov12"
CREATE_PREVIEW_FLAG="--create_preview"

echo "========================================================"
echo "🐟 直接分析鱼嘴行为任务开始"
echo "========================================================"
echo "作业ID: $SLURM_JOB_ID | 开始时间: $(date)"
echo "========================================================"

source ~/.bashrc
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR" || exit 1

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

echo ""
echo "========================================================"
echo "🚀 开始执行直接分析流水线..."
echo "========================================================"

START_TIME=$(date +%s)

eval python -u "$PYTHON_SCRIPT" \
    $PYTHON_INPUT_ARGS \
    --output_dir "\"$OUTPUT_DIR\"" \
    --head_model "\"$HEAD_MODEL_PATH\"" \
    --eye_model "\"$EYE_MODEL_PATH\"" \
    --mouth_model "\"$MOUTH_MODEL_PATH\"" \
    $CREATE_PREVIEW_FLAG \
    2>&1 | tee -a "$PROJECT_DIR/logs/analysis_${SLURM_JOB_ID}.log"

ANALYSIS_EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================================"
if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo "🎉 任务执行成功完成！"
else
    echo "❌ 任务执行失败，退出码: $ANALYSIS_EXIT_CODE"
fi
echo "总用时: $((DURATION / 3600))小时 $((DURATION % 3600 / 60))分钟 $((DURATION % 60))秒"
echo "========================================================"

exit $ANALYSIS_EXIT_CODE
