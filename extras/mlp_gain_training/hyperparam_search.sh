#!/bin/bash

set -e

# Hyperparameter grids
BATCH_SIZES=(64 128 256 512)
LEARNING_RATES=(0.01 0.001 0.0005 0.0001)
WEIGHT_DECAYS=(0.0 1e-5 1e-4 1e-3)

RESULT_LOG="result.log"
> "$RESULT_LOG"

for bs in "${BATCH_SIZES[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    for wd in "${WEIGHT_DECAYS[@]}"; do
      echo "Running batch_size=$bs lr=$lr weight_decay=$wd" | tee -a "$RESULT_LOG"
      python3 train_mlp_gain.py \
        --csv gru_training_data.csv \
        --out "run_bs${bs}_lr${lr}_wd${wd}" \
        --batch-size "$bs" \
        --fp32-lr "$lr" \
        --qat-lr "$lr" \
        --fp32-weight-decay "$wd" \
        --qat-weight-decay "$wd" \
        --dropout 0.1 \
        --seed 42 \
        > tmp.log
      tail -n 2 tmp.log >> "$RESULT_LOG"
      rm -f tmp.log
    done
  done
done
