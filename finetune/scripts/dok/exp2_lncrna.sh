#!/bin/bash
set -e

# nucleotide-transformer models
models=(
    "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    "InstaDeepAI/nucleotide-transformer-500m-1000g"
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g"
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
    "InstaDeepAI/agro-nucleotide-transformer-1b"
)

# 10 tasks
# (IA)^3 LR=3e-3 
# (3e-3, paper) Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning
for model in "${models[@]}"
do
    for seed in {40..44}
    do
        python train.py \
            --hf_model_path "$model" \
            --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
            --task_name "lncrna" \
            --output_dir "$SAKURA_ARTIFACT_DIR" \
            --project_name "$PROJECT_NAME" \
            --use_lora False \
            --use_ia3 True \
            --use_nt_kmer True \
            --per_device_train_batch_size "$BATCH_SIZE" \
            --per_device_eval_batch_size "$BATCH_SIZE" \
            --num_train_epochs 3 \
            --learning_rate 3e-3 \
            --warmup_ratio 0.1 \
            --save_strategy "epoch" \
            --evaluation_strategy "epoch" \
            --logging_steps 15000 \
            --fp16 True \
            --report_to "wandb" \
            --seed "$seed"
    done
done