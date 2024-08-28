#!/bin/bash

for data in terminator_strength promoter_strength, poly_a, lnc_rna, gene_exp, splicing
do
    # LoRA
    python train.py \
        --hf_model_path InstaDeepAI/agro-nucleotide-transformer-1b \
        --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
        --task_name data \
        --output_dir output \
        --project_name Plant-Molecular-Biology-2024 \
        --use_lora True \
        --lora_r 16 \
        --lora_alpha 8 \
        --lora_dropout 0.05 \
        --lora_target_modules "query,value" \
        --use_nt_kmer True \
        --kmer_window 6 \
        --kmer_stride 6 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs 4 \
        --learning_rate 1e-5 \
        --warmup_ratio 0.1 \
        --save_strategy "epoch" \
        --evaluation_strategy "epoch" \
        --logging_steps 1000 \
        --fp16 True \
        --report_to "wandb"
    
    # IA3
    python train.py \
        --hf_model_path InstaDeepAI/agro-nucleotide-transformer-1b \
        --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
        --task_name data \
        --output_dir output \
        --project_name Plant-Molecular-Biology-2024 \
        --use_ia3 True \
        --use_nt_kmer True \
        --kmer_window 6 \
        --kmer_stride 6 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs 4 \
        --learning_rate 1e-5 \
        --warmup_ratio 0.1 \
        --save_strategy "epoch" \
        --evaluation_strategy "epoch" \
        --logging_steps 1000 \
        --fp16 True \
        --report_to "wandb"

    # Baseline FT
    python train.py \
        --hf_model_path InstaDeepAI/agro-nucleotide-transformer-1b \
        --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
        --task_name data \
        --output_dir output \
        --project_name Plant-Molecular-Biology-2024 \
        --use_nt_kmer True \
        --kmer_window 6 \
        --kmer_stride 6 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --num_train_epochs 10 \
        --early_stopping_patience 3 \
        --learning_rate 2e-5 \
        --warmup_ratio 0.1 \
        --save_strategy "epoch" \
        --evaluation_strategy "epoch" \
        --logging_steps 1000 \
        --fp16 True \
        --report_to "wandb"
fi