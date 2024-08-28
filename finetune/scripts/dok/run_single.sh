#!/bin/bash
set -e

# sngle run
python train.py \
    --hf_model_path "$MODEL" \
    --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
    --task_name "$TASK" \
    --output_dir "$SAKURA_ARTIFACT_DIR" \
    --project_name "$PROJECT_NAME" \
    --use_lora "$USE_LORA" \
    --use_ia3 "$USE_IA3" \
    --lora_r "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --use_nt_kmer "$USE_NT_KMER" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --logging_steps 1000 \
    --fp16 True \
    --report_to "wandb"
