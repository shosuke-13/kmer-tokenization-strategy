import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Any

import wandb
import torch
import pandas as pd
import transformers
from loguru import logger
from peft import LoraConfig, IA3Config, get_peft_model, TaskType

from utils import generate_unique_run_name
from metrics import (
    calculate_r2_scores_by_experiment, 
    compute_metrics_for_classification, 
    compute_metrics_for_single_label_regression, 
    compute_metrics_for_multi_label_regression
)
from kmer import KmerTokenizer
from dataset import PlantGenomicBenchmark
from visualize import plot_averages, plot_tissue_specific, plot_expression_profiles


@dataclass
class ModelArguments:
    # Model configuration
    hf_model_path      : str  = field(
        default="InstaDeepAI/agro-nucleotide-transformer-1b",
        metadata={"help": "Path to the Hugging Face model"}
    )
    
    # LoRA configuration
    use_lora           : bool  = field(default=False, metadata={"help": "Whether to use LoRA"})
    lora_r             : int   = field(default=16, metadata={"help": "Hidden dimension for LoRA"})
    lora_alpha         : int   = field(default=8, metadata={"help": "Alpha for LoRA"})
    lora_dropout       : float = field(default=0.05, metadata={"help": "Dropout rate for LoRA"})
    lora_target_modules: str   = field(
        default="query,value",
        metadata={"help": "Where to perform LoRA (comma-separated list)"}
    )
    
    # IA3 configuration
    use_ia3            : bool = field(default=False, metadata={"help": "Whether to use IA3"})
    
    # Training configuration
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={"help": "Number of epochs with no improvement after which training will be stopped"}
    )

    # K-mer configuration
    use_nt_kmer        : bool = field(default=True, metadata={"help": "Whether to use nucleotide k-mers"})
    kmer_window        : int  = field(default=6, metadata={"help": "K-mer window size"})
    kmer_stride        : int  = field(default=6, metadata={"help": "K-mer stride"})


@dataclass
class DataArguments:
    # Dataset configuration
    hf_dataset_repo    : str  = field(
        default="InstaDeepAI/plant-genomic-benchmark",
        metadata={"help": "Hugging Face dataset repository"}
    )
    pgb_details_yaml   : str  = field(default="config/pgb_tasks.yaml", metadata={"help": "Path to the PGB details YAML file"})
    task_name          : str  = field(default="terminator_strength", metadata={"help": "Name of the task"})
    do_all_tasks       : bool = field(default=True, metadata={"help": "Whether to train on all tasks"})
    
    # Experiment tracking
    project_name       : str  = field(default="hf-peft", metadata={"help": "Weights & Biases project name"})
    
    # Data processing
    is_save_predictions: bool  = field(default=False, metadata={"help": "Whether to save predictions"})
    test_size          : float = field(default=0.1, metadata={"help": "Test size for train-test split"})
    val_split_seed     : int   = field(default=42, metadata={"help": "Seed for train-test split"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # General training settings
    cache_dir          : Optional[str] = field(default=None)
    run_name           : str  = field(default="run", metadata={"help": "Name of the training run"})
    optim              : str  = field(default="adamw_torch", metadata={"help": "Optimizer to use"})
    model_max_length   : int  = field(default=1024, metadata={"help": "Maximum sequence length"})
    
    # Batch and epoch settings
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size : int = field(default=1)
    num_train_epochs           : int = field(default=1)
    
    # Precision settings
    fp16               : bool = field(default=False, metadata={"help": "Whether to use 16-bit precision"})
    bf16               : bool = field(default=False, metadata={"help": "Whether to use 16-bit precision"})
    
    # Logging and evaluation settings
    logging_steps      : int  = field(default=1000)
    save_steps         : int  = field(default=1000)
    eval_steps         : int  = field(default=1000)
    evaluation_strategy: str  = field(default="steps")
    save_strategy      : str  = field(default="steps")
    
    # Optimization settings
    warmup_ratio       : float = field(default=0.1)
    weight_decay       : float = field(default=0.01)
    learning_rate      : float = field(default=1e-4)
    
    # Model saving settings
    save_total_limit      : int  = field(default=1)
    load_best_model_at_end: bool = field(default=True)
    output_dir            : str  = field(default="output")
    
    # Miscellaneous settings
    gradient_accumulation_steps: int = field(default=4)
    find_unused_parameters: bool = field(default=False)
    checkpointing         : bool = field(default=False)
    dataloader_pin_memory : bool = field(default=False)
    eval_and_save_results : bool = field(default=True)
    save_model            : bool = field(default=False)
    seed                  : int  = field(default=42)
    report_to             : str  = field(default=None)


def init_wandb(wandb_api_key: str, project_name: str, run_name: str) -> wandb.sdk.wandb_run.Run:
    if wandb_api_key:
        logger.info("Wandb API key found in environment variables.")

        wandb.login(key=wandb_api_key)
        run = wandb.init(project=project_name, name=run_name)

        logger.info(f"wandb run name: {run.name}")
        logger.success("wandb initialized successfully.")
        return run
    else:
        logger.warning("Wandb API key not found in environment variables. Skipping wandb initialization.")
        return None   


def save_results(pred, task_type: str, names: List[str]) -> pd.DataFrame:
    df = pd.DataFrame({"name": names})
    
    if task_type == "single_variable_regression":
        df["true_label"] = pred.label_ids
        df["pred_label"] = pred.predictions.squeeze()
    elif task_type == "multi_variable_regression":
        true_labels = pred.label_ids
        pred_labels = pred.predictions

        for i in range(pred_labels.shape[1]):
            df[f"true_label_{i}"] = true_labels[:, i]
            df[f"pred_label_{i}"] = pred_labels[:, i]
    else:
        # binary classification
        df["true_label"] = pred.label_ids
        df["pred_label"] = pred.predictions.argmax(axis=1)
        df["pred_score_0"] = pred.predictions[:, 0]
        df["pred_score_1"] = pred.predictions[:, 1]
    
    return df


def log_metrics_to_wandb(
        run: wandb.sdk.wandb_run.Run,
        results: Dict[str, Any],
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments
    ) -> None:
    """Log metrics to Weights & Biases (wandb)."""

    metrics = {
        "model_name"      : model_args.hf_model_path,
        "task_name"       : data_args.task_name,
        "epochs"          : training_args.num_train_epochs,
        "learning_rate"   : training_args.learning_rate,
        "train_batch_size": training_args.per_device_train_batch_size,
        "eval_batch_size" : training_args.per_device_eval_batch_size,
        "kmer_window"     : data_args.kmer_window,
        "kmer_stride"     : data_args.kmer_stride,
    }

    if model_args.use_lora:
        logger.debug("Logging LoRA metrics")
        metrics.update({
            "use_lora"           : model_args.use_lora,
            "lora_r"             : model_args.lora_r,
            "lora_alpha"         : model_args.lora_alpha,
            "lora_dropout"       : model_args.lora_dropout,
            "lora_target_modules": model_args.lora_target_modules
        })
    elif model_args.use_ia3:
        logger.debug("Logging IA3 metrics")
        metrics.update({
            "use_ia3"                : model_args.use_ia3,
            "ia3_target_modules"     : "key, value, intermediate.dense",
            "ia3_feedforward_modules": "intermediate.dense"
        })
    else:
        logger.debug("Logging fine-tuning metrics")

    metrics.update(results)
    run.log({"metrics": wandb.Table(dataframe=pd.DataFrame([metrics]))})



def load_and_split_dataset(task_name: str, tokenizer, model_args: ModelArguments, training_arg: TrainingArguments):
    pgb = PlantGenomicBenchmark(pgb_config="config/pgb_tasks.yaml", expression_config="config/expression_tasks.yaml")
    dataset = pgb.load_dataset(task_name)

    # kmer tokenization
    num_tokens = pgb.get_num_tokens(
        max_seq_len=pgb.get_max_seq_length(task_name), 
        kmer_window=model_args.kmer_window, 
        kmer_stride=model_args.kmer_stride, 
        model_max_length=training_arg.model_max_length
    )
    tokenized_datasets = pgb.tokenize_dataset(dataset, tokenizer, num_tokens)

    return pgb.split_dataset(tokenized_datasets) # train, val, test


def get_model(task_type: str, num_labels: int, model_args: ModelArguments, training_args: TrainingArguments):
    if task_type == "binary_classification":
         return transformers.AutoModelForSequenceClassification.from_pretrained(
                model_args.hf_model_path,
                cache_dir=training_args.cache_dir,
                num_labels=num_labels,
                trust_remote_code=True,
            )
    else:
        # single_variable_regression or multi_variable_regression
        return transformers.AutoModelForSequenceClassification.from_pretrained(
                model_args.hf_model_path,
                num_labels=num_labels,
                problem_type="regression",
                trust_remote_code=True,
            )


def set_peft_model(model, model_args: ModelArguments):
    if model_args.use_lora:
        lora_config = LoraConfig(
                        r=model_args.lora_r,
                        lora_alpha=model_args.lora_alpha,
                        target_modules=list(model_args.lora_target_modules.split(",")),
                        lora_dropout=model_args.lora_dropout,
                        bias="none",
                        task_type=TaskType.SEQ_CLS,
                        inference_mode=False,
                    )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif model_args.use_ia3:
        ia3_config = IA3Config(
                        task_type=TaskType.SEQ_CLS,
                        target_modules=["key", "value", "intermediate.dense"],
                        feedforward_modules=["intermediate.dense"],
                    )
        model = get_peft_model(model, ia3_config)
        model.print_trainable_parameters()


def train(
        model_args: ModelArguments, 
        data_args: DataArguments, 
        training_args: TrainingArguments, 
        tokenizer: transformers.PreTrainedTokenizer, 
        task_type: str,
        run: Optional[wandb.sdk.wandb_run.Run] = None
    ) -> None:

    # the number of prediction labels
    if task_type == "binary_classification":
        num_labels = 2
        compute_metrics = compute_metrics_for_classification
    elif task_type == "multi_variable_regression":
        num_labels = len(train_dataset[0]["labels"])
        compute_metrics = compute_metrics_for_multi_label_regression
    elif task_type == "single_variable_regression":
        num_labels = 1
        compute_metrics = compute_metrics_for_single_label_regression
    else:
        logger.error(f"Unsupported task type: {task_type}")

    # load plant-genomic-benchmark dataset
    train_dataset, eval_dataset, test_dataset = load_and_split_dataset(
        task_name=data_args.task_name, 
        tokenizer=tokenizer, 
        model_args=model_args, 
        training_arg=training_args
    )

    model = get_model(task_type, num_labels, model_args, training_args)
    peft_model = set_peft_model(model, model_args, training_args)

    # define trainer
    trainer = transformers.Trainer(
                model=peft_model,
                tokenizer=tokenizer,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=transformers.EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping_patience),
            )
    
    # finetune
    trainer.train()
    trainer.save_model(training_args.output_dir)

    results = trainer.evaluate(eval_dataset=test_dataset)
    pred = trainer.predict(test_dataset)

    if data_args.task_name.endswith("leaf") or data_args.task_name.endswith("protoplast"):
        results.update(calculate_r2_scores_by_experiment(pred, test_dataset))

    if data_args.task_name.startswith("gene_exp"):
        save_expression_results(pred, training_args.output_dir, data_args.task_name.split(".")[-1])

    return results, pred


def save_expression_results(pred, output_dir, task_name):
    pgb = PlantGenomicBenchmark()
    tissues, num_rows, num_cols = pgb.get_tissue_name(task_name)

    # expression profile plots
    plot_averages(pred, output_dir)
    plot_tissue_specific(pred, output_dir, num_rows, num_cols, tissues)
    plot_expression_profiles(
        pred, 
        tissues, 
        output_dir, 
        mode='side_by_side', 
        cmap_list=['Blues', 'Purples', 'Greens', 'Reds']
    )
    plot_expression_profiles(
        pred, 
        tissues, 
        output_dir, 
        mode='separate', 
        cmap_list=['Blues', 'Purples', 'Greens', 'Reds']
    )

    wandb.log({"average_plot": wandb.Image(f"{output_dir}/prediction_actual_plot_all_tissues.png")})
    wandb.log({"tissue_specific_plot": wandb.Image(f"{output_dir}/prediction_actual_plots_tissues.png")})

    for cmap_color in ['Blues', 'Purples', 'Greens', 'Reds']:
        wandb.log({
            "expression_profiles_side_by_side": 
            wandb.Image(f"{output_dir}/tissue_expression_profiles_comparison_{cmap_color}.png")
        })
        wandb.log({
            "true_expression_profiles": 
            wandb.Image(f"{output_dir}/true_expression_profiles_{cmap_color}.png"),
            "predicted_expression_profiles": 
            wandb.Image(f"{output_dir}/predicted_expression_profiles_{cmap_color}.png")
        })


def main():
    """fine-tune Huggingface DNA foundation models."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logger.add(os.path.join(training_args.output_dir, "train_and_evaulate.log"), rotation="10 MB")

    # load tokenizer (DNABERT-based or NT-based)
    if data_args.use_nt_kmer:
        # NT, NT-V2 and AgroNT uses the same tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained("InstaDeepAI/agro-nucleotide-transformer-1b")
    else:
        tokenizer =  KmerTokenizer(
                vocab_file=f"kmer/kmer_{data_args.kmer_window}/vocab.txt",
                max_length=512,
                model_max_length=512,
                k=data_args.kmer_window,
                stride=data_args.kmer_stride,
            )
    
    # fine-tune on PGB dataset
    wandb_api_key = os.environ.get("WANDB_API_KEY")

    pgb = PlantGenomicBenchmark()
    use_peft = "lora" if model_args.use_lora else "ia3" if model_args.use_ia3 else "ft"

    if data_args.do_all_tasks: # train on all tasks
        for detail_name in pgb.pgb_config["tasks"]:
            task_name = data_args.task_name
            data_args.task_name = data_args.task_name + "." + detail_name
            
            # initialize wandb
            run_name = generate_unique_run_name(model_args.hf_model_path, data_args.task_name, use_peft)
            current_run = init_wandb(wandb_api_key, data_args.project_name, run_name)
            
            results, pred = train(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                tokenizer=tokenizer,
                task_type=pgb.pgb_config[task_name]["type"],
                run=current_run
            )
            
            # log evaluation metrics
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(os.path.join(training_args.output_dir, "eval_results.json"), "w") as f:
                json.dump(results, f)
            
            log_metrics_to_wandb(run, results, model_args, data_args, training_args)

            data_args.task_name = data_args.task_name.split(".")[0] # reset task_name
            wandb.finish()
    else:
        # train on single task
        run_name = generate_unique_run_name(model_args.hf_model_path, data_args.task_name, use_peft)
        run = init_wandb(wandb_api_key, data_args.project_name, run_name)

        results, pred = train(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            tokenizer=tokenizer,
            task_type=pgb.pgb_config[data_args.task_name]["type"],
            run=run
        )

        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, "eval_results.json"), "w") as f:
            json.dump(results, f)
        
        log_metrics_to_wandb(run, results, model_args, data_args, training_args)
        wandb.finish()


if __name__ == "__main__":
    main()
