#!/usr/bin/env bash
# run_evaluate.sh — 一鍵執行 AudioMOS 推論與評估

export CUDA_VISIBLE_DEVICES=0

# —— 參數設定 ——
POOLING_TYPE=drsp
SEGMENT_SIZE=4

EXPNAME="track2_${POOLING_TYPE}_s${SEGMENT_SIZE}"

OUTPUT_CSV="../evaluation/"$EXPNAME"/answer.txt"
CKPT="./track2_ckpt/"$EXPNAME"/best_model_24.pt"

# Create the output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_CSV")"

python evaluate.py \
  --eval_list /share/nas169/jerryyang/AudioMOS/track2/audiomos2025-track2-eval-phase/DATA/sets/eval_list.txt \
  --wav_dir /share/nas169/jerryyang/AudioMOS/track2/audiomos2025-track2-eval-phase/DATA/wav \
  --output_csv "$OUTPUT_CSV" \
  --ckpt "$CKPT" \
  --batch_size 1 \
  --pooling_type  "$POOLING_TYPE" \
  --segment_size  "$SEGMENT_SIZE"

