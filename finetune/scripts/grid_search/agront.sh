#!/bin/bash
set -e
OUTPUT_DIR="output"

for data in terminator_strength promoter_strength, poly_a, lnc_rna, gene_exp, splicing
do
    # LoRA
    for epoch in 2 3 4
    do
        for lr in 1e-4 2e-4 1e-5
        do
            for lora_weight in "query,value" "query,key,value,output"
            do
                for lora_r in 8 16 32
                do
                    for lora_alpha in 4 8 16
                    do
                        python train.py \
                            --hf_model_path InstaDeepAI/agro-nucleotide-transformer-1b \
                            --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
                            --task_name data \
                            --output_dir "$OUTPUT_DIR" \
                            --project_name Plant-Molecular-Biology-2024 \
                            --use_lora True \
                            --lora_r $lora_r \
                            --lora_alpha $lora_alpha \
                            --lora_dropout 0.05 \
                            --lora_target_modules $lora_weight \
                            --use_nt_kmer True \
                            --kmer_window 6 \
                            --kmer_stride 6 \
                            --per_device_train_batch_size 32 \
                            --per_device_eval_batch_size 32 \
                            --num_train_epochs 4 \
                            --learning_rate $lr \
                            --warmup_ratio 0.1 \
                            --save_strategy "epoch" \
                            --evaluation_strategy "epoch" \
                            --logging_steps 10000 \
                            --fp16 True \
                            --report_to "wandb"
                    done
                done
            done
        done
    done
    
    # IA3
    for epoch in 2 3 4
    do
        for lr in 3e-3 1e-4
        do
            python train.py \
                --hf_model_path InstaDeepAI/agro-nucleotide-transformer-1b \
                --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
                --task_name data \
                --output_dir "$OUTPUT_DIR" \
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
                --logging_steps 10000 \
                --fp16 True \
                --report_to "wandb"
        done
    done

    # Baseline FT
    python train.py \
        --hf_model_path InstaDeepAI/agro-nucleotide-transformer-1b \
        --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
        --task_name data \
        --output_dir "$OUTPUT_DIR" \
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
        --logging_steps 10000 \
        --fp16 True \
        --report_to "wandb"
done
