#!/bin/bash
OUTPUT_DIR=$SAKURA_ARTIFACT_DIR

models=(
    "InstaDeepAI/nucleotide-transformer-500m-1000g"
    "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g"
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
)

run_names=(
    "NT_500_1000g"
    "NT_500_human"
    "NT_2500_1000g"
    "NT_2500_multi"
)

current_model=$2

if [ "$current_model" -ge 0 ] && [ "$current_model" -lt ${#models[@]} ]; then
    model=${models[$current_model]}
    run_name=${run_names[$current_model]}
    echo "current nt version: $model"
else
    echo "Wrong argument. Please choose a number between 0 and $((${#models[@]} - 1))."
    exit 1
fi

# Run the Python script
for data in terminator_strength promoter_strength, poly_a, lnc_rna, gene_exp, splicing
do
    # LoRA
    for lr in 1e-5 2e-5
    do
        for lora_weight in "query,value" "query,key,value,output"
            do
            python train.py \
                --hf_model_path InstaDeepAI/agro-nucleotide-transformer-1b \
                --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
                --task_name data \
                --output_dir $OUTPUT_DIR \
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
                --logging_steps 10000 \
                --fp16 True \
                --report_to "wandb"
            done
        done
    done
    
    # IA3
    for lr in 1e-3 1e-4
    do
        python train.py \
            --hf_model_path InstaDeepAI/agro-nucleotide-transformer-1b \
            --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
            --task_name data \
            --output_dir $OUTPUT_DIR \
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

    # Baseline FT
    python train.py \
        --hf_model_path InstaDeepAI/agro-nucleotide-transformer-1b \
        --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
        --task_name data \
        --output_dir $OUTPUT_DIR \
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
