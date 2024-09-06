#!/bin/bash
set -e

# plant-genomic-benchmark
tasks=(
    "promoter_strength"
    "terminator_strength"
    "poly_a"
)

# nucleotide-transformer models
models=(
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g"
    "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    "InstaDeepAI/nucleotide-transformer-500m-1000g"
    "InstaDeepAI/agro-nucleotide-transformer-1b"
)

# 80 tasks
for model in "${models[@]}"
do
    for task in "${tasks[@]}"
    do
        python train.py \
            --hf_model_path "$model" \
            --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
            --task_name "$task" \
            --output_dir "$SAKURA_ARTIFACT_DIR" \
            --project_name "$PROJECT_NAME" \
            --use_lora True \
            --use_ia3 False \
            --use_nt_kmer True \
            --per_device_train_batch_size "$BATCH_SIZE" \
            --per_device_eval_batch_size "$BATCH_SIZE" \
            --num_train_epochs "$EPOCHS" \
            --learning_rate 1e-4 \
            --warmup_ratio 0.1 \
            --save_strategy "epoch" \
            --evaluation_strategy "epoch" \
            --logging_steps 15000 \
            --fp16 True \
            --report_to "wandb" \
            --seed "$SEED"
    done
done
