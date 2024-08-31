#!/bin/bash
set -e

# overlap
for k in 3 4 5 6
do
    python train.py \
        --hf_model_path zhihan1996/DNA_bert_"$k" \
        --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
        --task_name "$TASK" \
        --output_dir "$SAKURA_ARTIFACT_DIR" \
        --project_name "$PROJECT_NAME" \
        --use_nt_kmer "$USE_NT_KMER" \
        --kmer_window "$k" \
        --kmer_stride 1 \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --num_train_epochs 10 \
        --early_stopping_patience "$EARLY_STOP" \
        --learning_rate "$LR" \
        --warmup_ratio 0.1 \
        --save_strategy "epoch" \
        --evaluation_strategy "epoch" \
        --logging_steps 1000 \
        --fp16 True \
        --report_to "wandb"
done

# non-overlap
for k in 3 4 5 6
do
    python train.py \
        --hf_model_path zhihan1996/DNA_bert_"$k" \
        --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
        --task_name "$TASK" \
        --output_dir "$SAKURA_ARTIFACT_DIR" \
        --project_name "$PROJECT_NAME" \
        --use_nt_kmer "$USE_NT_KMER" \
        --kmer_window "$k" \
        --kmer_stride "$k" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --num_train_epochs 10 \
        --early_stopping_patience "$EARLY_STOP" \
        --learning_rate "$LR" \
        --warmup_ratio 0.1 \
        --save_strategy "epoch" \
        --evaluation_strategy "epoch" \
        --logging_steps 1000 \
        --fp16 True \
        --report_to "wandb"
done
