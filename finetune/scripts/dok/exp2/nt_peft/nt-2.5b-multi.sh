#!/bin/bash
set -e

for seed in 42 123 256
do
    # LoRA
    python train.py \
        --hf_model_path "InstaDeepAI/nucleotide-transformer-2.5b-multi-species" \
        --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
        --task_name "$TASK" \
        --output_dir "$SAKURA_ARTIFACT_DIR" \
        --project_name "$PROJECT_NAME" \
        --use_lora True \
        --use_ia3 False \
        --lora_r 16 \
        --lora_alpha 8 \
        --lora_dropout 0.05 \
        --lora_target_modules "query,key,value,output.dense" \
        --use_nt_kmer True \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --num_train_epochs "$EPOCHS" \
        --learning_rate 2e-4 \
        --warmup_ratio 0.1 \
        --save_strategy "epoch" \
        --evaluation_strategy "epoch" \
        --logging_steps 1000 \
        --fp16 True \
        --report_to "wandb" \
        --seed "$seed"


    # IA3
    python train.py \
        --hf_model_path "InstaDeepAI/nucleotide-transformer-2.5b-multi-species" \
        --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
        --task_name "$TASK" \
        --output_dir "$SAKURA_ARTIFACT_DIR" \
        --project_name "$PROJECT_NAME" \
        --use_lora False \
        --use_ia3 True \
        --use_nt_kmer True \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --num_train_epochs "$EPOCHS" \
        --learning_rate 1e-3 \
        --warmup_ratio 0.1 \
        --save_strategy "epoch" \
        --evaluation_strategy "epoch" \
        --logging_steps 1000 \
        --fp16 True \
        --report_to "wandb" \
        --seed "$seed"
done