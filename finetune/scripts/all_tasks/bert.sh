#!/bin/bash

for data in terminator_strength promoter_strength, poly_a, lnc_rna, gene_exp, splicing
do  
    # overlap
    for k in 3,4,5,6,7,8
    do
        python train.py \
            --hf_model_path suke-sho/BERT-K$k \
            --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
            --task_name data \
            --output_dir output \
            --project_name Plant-Molecular-Biology-2024 \
            --use_nt_kmer True \
            --kmer_window $k \
            --kmer_stride 1 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --num_train_epochs 10 \
            --early_stopping_patience 3 \
            --learning_rate 2e-5 \
            --warmup_ratio 0.1 \
            --save_strategy "epoch" \
            --evaluation_strategy "epoch" \
            --logging_steps 1000 \
            --fp16 True \
            --report_to "wandb"
    done

    # non-overlap
    for k in 3,4,5,6,7,8
    do
        python train.py \
            --hf_model_path suke-sho/BERT-K4$k \
            --hf_dataset_repo InstaDeepAI/plant-genomic-benchmark \
            --task_name data \
            --output_dir output \
            --project_name Plant-Molecular-Biology-2024 \
            --use_nt_kmer True \
            --kmer_window $k \
            --kmer_stride $k \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --num_train_epochs 10 \
            --early_stopping_patience 3 \
            --learning_rate 2e-5 \
            --warmup_ratio 0.1 \
            --save_strategy "epoch" \
            --evaluation_strategy "epoch" \
            --logging_steps 1000 \
            --fp16 True \
            --report_to "wandb"
    done
done
