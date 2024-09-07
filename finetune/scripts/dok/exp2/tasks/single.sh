#!/bin/bash
set -e

# plant-genomic-benchmark
tasks=(
    "promoter_strength"
    "terminator_strength"
    "poly_a"
)

for task in "${tasks[@]}"
do
    python train.py \
        --hf_model_path "$MODEL" \
        --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
        --task_name "$task" \
        --output_dir "$SAKURA_ARTIFACT_DIR" \
        --project_name "$PROJECT_NAME" \
        --use_lora True \
        --use_ia3 False \
        --use_nt_kmer True \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --num_train_epochs 3 \
        --learning_rate 1e-4 \
        --warmup_ratio 0.1 \
        --save_strategy "epoch" \
        --evaluation_strategy "epoch" \
        --logging_steps 15000 \
        --fp16 True \
        --report_to "wandb" \
        --seed "$SEED" \
        --is_save_predictions True \
        --gradient_accumulation_steps 4
done
