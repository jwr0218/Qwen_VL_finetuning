#!/bin/bash
export CODECARBON_LOG_LEVEL="error"
export CODECARBON_DISABLED="true"  # 탄소 측정이 필요 없다면 아예 끄기

# --- 기본 설정 ---
BASE_MODEL="huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"
DATA_PATH="/workspace/Toonspace_VLM/data/ocr_description/total(bbox_normal)_ocr_dataset_2F.json"
PROJECT_NAME="qwen3-Native-Stepped-Training"
OUTPUT_ROOT="ex_models/native_stepped"

# --- Step별 하이퍼파라미터 설정 ---

# [Step 1] Internal Adapter: 초기 정렬을 위해 1:1 비율 사용
STEP1_LR=1e-4
STEP1_TEXT_WEIGHT=1.0
STEP1_BBOX_WEIGHT=1.0

# [Step 2] Decoder: 텍스트 생성 능력 강화 (기존 설정 복귀)
STEP2_LR=2e-5
STEP2_TEXT_WEIGHT=2.0
STEP2_BBOX_WEIGHT=0.1

# [Step 3] Full Fine-tuning: 전체 최적화
STEP3_LR=5e-6
STEP3_TEXT_WEIGHT=2.0
STEP3_BBOX_WEIGHT=0.1

# 공통 설정
BATCH_SIZE=1
ACCUM_STEPS=4

# 에러 발생 시 중단
set -e

# 출력 디렉토리 정의
STEP1_DIR="${OUTPUT_ROOT}/step1_internal_adapter"
STEP2_DIR="${OUTPUT_ROOT}/step2_decoder"
STEP3_DIR="${OUTPUT_ROOT}/step3_full"

# echo "======================================================"
# echo "🚀 Step 1: Internal Adapter Only (Weight ${STEP1_TEXT_WEIGHT}:${STEP1_BBOX_WEIGHT})"
# echo "======================================================"

# accelerate launch --multi_gpu --num_processes=2 train/step_train.py \
#     --model_id "$BASE_MODEL" \
#     --processor_id "$BASE_MODEL" \
#     --data_path "$DATA_PATH" \
#     --output_dir "$STEP1_DIR" \
#     --stage "step1" \
#     --wandb_project "$PROJECT_NAME" \
#     --wandb_name "step1_merger_only" \
#     --num_train_epochs 3 \
#     --learning_rate $STEP1_LR \
#     --per_device_train_batch_size $BATCH_SIZE \
#     --gradient_accumulation_steps $ACCUM_STEPS \
#     --text_weight $STEP1_TEXT_WEIGHT \
#     --bbox_weight $STEP1_BBOX_WEIGHT \

# echo "✅ Step 1 Completed."
# sleep 60

# echo "======================================================"
# echo "🚀 Step 2: Decoder Training (Weight ${STEP2_TEXT_WEIGHT}:${STEP2_BBOX_WEIGHT})"
# echo "======================================================"

# accelerate launch --multi_gpu --num_processes=2 train/step_train.py \
#     --model_id "$BASE_MODEL" \
#     --processor_id "$BASE_MODEL" \
#     --data_path "$DATA_PATH" \
#     --resume_from_prev_stage "$STEP1_DIR" \
#     --output_dir "$STEP2_DIR" \
#     --stage "step2" \
#     --wandb_project "$PROJECT_NAME" \
#     --wandb_name "step2_decoder_ft" \
#     --num_train_epochs 5 \
#     --learning_rate $STEP2_LR \
#     --per_device_train_batch_size $BATCH_SIZE \
#     --gradient_accumulation_steps $ACCUM_STEPS \
#     --text_weight $STEP2_TEXT_WEIGHT \
#     --bbox_weight $STEP2_BBOX_WEIGHT \

# echo "✅ Step 2 Completed."
# sleep 60

echo "======================================================"
echo "🚀 Step 3: Full Fine-tuning (Weight ${STEP3_TEXT_WEIGHT}:${STEP3_BBOX_WEIGHT})"
echo "======================================================"

accelerate launch --multi_gpu --num_processes=2 train/step_train.py \
    --model_id "$BASE_MODEL" \
    --processor_id "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --resume_from_prev_stage "$STEP2_DIR" \
    --output_dir "$STEP3_DIR" \
    --stage "step3" \
    --wandb_project "$PROJECT_NAME" \
    --wandb_name "step3_full_ft" \
    --num_train_epochs 7 \
    --learning_rate $STEP3_LR \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --text_weight $STEP3_TEXT_WEIGHT \
    --bbox_weight $STEP3_BBOX_WEIGHT \

echo "🎉 All Steps Completed Successfully!"