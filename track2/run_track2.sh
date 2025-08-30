#!/usr/bin/env bash
# run_track2.sh — 一鍵執行 AudioMOS Track2 訓練

export CUDA_VISIBLE_DEVICES=0
# export WANDB_MODE=disabled

# —— 參數設定 —— 
TRAIN_CSV="../audiomos2025_track2/audiomos2025-track2-train_list_filtered.csv"
DEV_CSV="../audiomos2025_track2/audiomos2025-track2-dev_list_filtered.csv"

EPOCHS=1000
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=8
LR=1e-4
FREEZE_ENCODER=1

POOLING_TYPE=drsp
SEGMENT_SIZE=4

EXPNAME="track2_${POOLING_TYPE}_s${SEGMENT_SIZE}"


# —— 執行 train script —— 
python mos_track2.py \
  --train_csv  "$TRAIN_CSV" \
  --dev_csv    "$DEV_CSV"   \
  --exp_name   "$EXPNAME"       \
  --epochs     $EPOCHS      \
  --train_batch_size $TRAIN_BATCH_SIZE       \
  --eval_batch_size  $EVAL_BATCH_SIZE       \
  --lr         $LR          \
  --freeze_encoder  $FREEZE_ENCODER \
  --pooling_type    $POOLING_TYPE \
  --segment_size    $SEGMENT_SIZE