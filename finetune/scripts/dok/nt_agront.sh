#!/bin/bash
set -e

# LoRA
for lr in 1e-5 2e-5
do
    for lora_weight in "query,value" "query,key,value,output"
    do
    python train.py \
        --hf_model_path "$MODEL" \
        --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
        --task_name "$TASK" \
        --output_dir "$SAKURA_ARTIFACT_DIR" \
        --project_name "$PROJECT_NAME" \
        --use_lora True \
        --use_ia3 False \
        --lora_r "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout 0.05 \
        --lora_target_modules "$lora_weight" \
        --use_nt_kmer "$USE_NT_KMER" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --num_train_epochs "$EPOCHS" \
        --learning_rate "$lr" \
        --warmup_ratio 0.1 \
        --save_strategy "epoch" \
        --evaluation_strategy "epoch" \
        --logging_steps 1000 \
        --fp16 True \
        --report_to "wandb"
    done
done

# IA3
for lr in 1e-3 1e-4
do
    python train.py \
        --hf_model_path "$MODEL" \
        --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
        --task_name "$TASK" \
        --output_dir "$SAKURA_ARTIFACT_DIR" \
        --project_name "$PROJECT_NAME" \
        --use_lora False \
        --use_ia3 True \
        --use_nt_kmer "$USE_NT_KMER" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --num_train_epochs "$EPOCHS" \
        --learning_rate "$lr" \
        --warmup_ratio 0.1 \
        --save_strategy "epoch" \
        --evaluation_strategy "epoch" \
        --logging_steps 1000 \
        --fp16 True \
        --report_to "wandb"
done
