#!/bin/bash
set -e

# nucleotide-transformer models
models=(
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g"
    "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    "InstaDeepAI/nucleotide-transformer-500m-1000g"
    "InstaDeepAI/agro-nucleotide-transformer-1b"
)

# (IA)^3 LR=3e-3 
# (3e-3, paper) Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning
for model in "${models[@]}"
do
    python train.py \
            --hf_model_path "$model" \
            --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
            --task_name "gene_exp" \
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
            --seed 42 \
            --is_save_predictions True
done
