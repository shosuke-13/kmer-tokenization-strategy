import os
import sys
import json
import time
import yaml
import math
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Any

import wandb
import torch
import numpy as np
import pandas as pd
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    precision_recall_curve,
    auc,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from peft import (
    LoraConfig,
    IA3Config,
    get_peft_model,
    TaskType,
)
from loguru import logger

from kmer import KmerTokenizer


from dataclasses import dataclass, field
from typing import Optional, List
import transformers


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


@dataclass
class DataArguments:
    # Dataset configuration
    hf_dataset_repo    : str  = field(
        default="InstaDeepAI/plant-genomic-benchmark",
        metadata={"help": "Hugging Face dataset repository"}
    )
    pgb_details_yaml   : str  = field(default="pgb_tasks.yaml", metadata={"help": "Path to the PGB details YAML file"})
    task_name          : str  = field(default="terminator_strength", metadata={"help": "Name of the task"})
    do_all_tasks       : bool = field(default=True, metadata={"help": "Whether to train on all tasks"})

    # K-mer configuration
    use_nt_kmer        : bool = field(default=True, metadata={"help": "Whether to use nucleotide k-mers"})
    kmer_window        : int  = field(default=6, metadata={"help": "K-mer window size"})
    kmer_stride        : int  = field(default=6, metadata={"help": "K-mer stride"})
    
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
    model_max_length   : int  = field(default=512, metadata={"help": "Maximum sequence length"})
    
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
    find_unused_parameters: bool = field(default=False)
    checkpointing         : bool = field(default=False)
    dataloader_pin_memory : bool = field(default=False)
    eval_and_save_results : bool = field(default=True)
    save_model            : bool = field(default=False)
    seed                  : int  = field(default=42)
    report_to             : str  = field(default=None)


def tokenize_function(tokenizer, sample: Dict[str, Any], max_length: int) -> Dict[str, Any]:
    """Tokenizes single sequence."""
    sequence = sample["sequence"]
    encoded = tokenizer(
        sequence,
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": sample["label"],
    }


def pgb_dataset_details(data_args: DataArguments) -> Dict[str, Any]:
    try:
        with open(data_args.pgb_details_yaml, 'r') as file:
            config = yaml.safe_load(file)
            pgb_details = config['pgb_tasks']
    except FileNotFoundError:
        logger.error("pgb_tasks.yaml file not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        sys.exit(1)
    
    if data_args.do_all_tasks:
        return pgb_details[data_args.task_name]
    else:
        # single task given full task name, e.g., "terminator_strength.leaf"
        return pgb_details[data_args.task_name.split(".")[0]]


def pgb_dataset(
        data_args: DataArguments, 
        training_args: ModelArguments, 
        tokenizer: transformers.PreTrainedTokenizer, 
        max_seq_len: int
    ) -> Tuple[Dataset, Dataset, Dataset]:
    """Load plant-genomic-benchmark dataset and tokenize."""
    try:
        logger.debug(f"Hugging Face dataset repository: {data_args.hf_dataset_repo}")
        logger.debug(f"Task name: {data_args.task_name}")

        # BERT/DNABERT: 512 tokens, NT/AgroNT: 1024 tokens
        num_tokens = min(math.ceil((max_seq_len - data_args.kmer_window + 1) / data_args.kmer_stride), training_args.model_max_length)
        logger.debug(f"Max number of tokens: {num_tokens}")

        dataset = load_dataset(data_args.hf_dataset_repo, task_name=data_args.task_name, trust_remote_code=True)
        tokenized_datasets = dataset.map(
            lambda x: tokenize_function(tokenizer, x, num_tokens),
            batched=True
        )

        logger.success(f"Dataset loaded and tokenized successfully.")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)

    if "validation" in tokenized_datasets.keys():
        logger.debug("validation set found.")
        return (
            tokenized_datasets["train"],
            tokenized_datasets["validation"],
            tokenized_datasets["test"],
        )
    else:
        logger.debug("validation set not found.")
        train_datasets = tokenized_datasets["train"].train_test_split(test_size=data_args.test_size, seed=data_args.val_split_seed)
        return train_datasets["train"], train_datasets["test"], tokenized_datasets["test"]


def get_compute_metrics(task_type: str):
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1) if task_type == "classification" else pred.predictions.squeeze()

        try:
            if task_type == "classification":
                precision, recall, _ = precision_recall_curve(labels, preds)
                return {
                    "accuracy": accuracy_score(labels, preds),
                    "f1": f1_score(labels, preds),
                    "precision": precision_score(labels, preds),
                    "recall": recall_score(labels, preds),
                    "mcc": matthews_corrcoef(labels, preds),
                    "roc_auc": roc_auc_score(labels, preds),
                    "pr_auc": auc(recall, precision),
                }
            else:
                return {
                    "mse": mean_squared_error(labels, preds),
                    "mae": mean_absolute_error(labels, preds),
                    "r2": r2_score(labels, preds),
                }
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            sys.exit(1)
    return compute_metrics


def save_results(predictions, task_type: str, out_dir: str, names: List[str]) -> None:
    """Save observed and predicted values."""
    df = pd.DataFrame({"name": names})

    # binary classification and single-label regression
    df["true_label"] = predictions.true_labels
    df["pred_label"] = predictions.pred_labels

    if task_type == "classification":
        # binary classification
        df["pred_score_0"] = predictions[:, 0]
        df["pred_score_1"] = predictions[:, 1]
    if len(df["true_labels"].shape) > 1:
        # multi-label regression
        for i in range(predictions.true_labels.shape[1]):
            df[f"true_label_{i}"] = predictions.true_labels[:, i]
            df[f"pred_label_{i}"] = predictions.pred_labels[:, i]

    df.to_parquet(os.path.join(out_dir, "predictions.parquet"), index=False)


def generate_unique_run_name(hf_model_path: str, task_name: str) -> str:
    """unique run name for tracking experiments."""
    timestamp = int(time.time())
    hash_input = f"{hf_model_path}_{task_name}_{timestamp}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    model_name = hf_model_path.split("/")[-1]
    return f"{model_name}_{task_name}_{hash_value}"


def get_callbacks(model_args):
    """Early stopping callback if full fine-tuning."""
    callbacks = []
    if model_args.early_stopping_patience is not None and model_args.early_stopping_patience > 0:
        callbacks.append(transformers.EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping_patience))
    return callbacks


def train(
        model_args: ModelArguments, 
        data_args: DataArguments, 
        training_args: TrainingArguments, 
        tokenizer: transformers.PreTrainedTokenizer, 
        task_details: Dict[str, Any]
    ) -> None:
    """Train and evaluate model."""

    train_dataset, eval_dataset, test_dataset = pgb_dataset(
        data_args=data_args,
        training_args=training_args,
        tokenizer=tokenizer,
        max_seq_len=task_details["max_seq_len"]
    )

    # load model
    try:
        logger.debug(f"Load {task_details['type']} model: num_labels={task_details['num_labels']}")
        if task_details["type"] == "classification":
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                        model_args.hf_model_path,
                        cache_dir=training_args.cache_dir,
                        num_labels=task_details["num_labels"],
                        trust_remote_code=True,
                    )
        else:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                        model_args.hf_model_path,
                        num_labels=task_details["num_labels"],
                        problem_type="regression",
                        trust_remote_code=True
                    )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    # configure peft
    try:
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

            logger.success(f"LoRA configured successfully.")
        elif model_args.use_ia3:
            ia3_config = IA3Config(
                            task_type=TaskType.SEQ_CLS,
                            target_modules=["key", "value", "intermediate.dense"],
                            feedforward_modules=["intermediate.dense"],
                        )
            model = get_peft_model(model, ia3_config)
            model.print_trainable_parameters()

            logger.success(f"IA3 configured successfully.")
        elif model_args.use_lora and model_args.use_ia3:
            logger.error("Both LoRA and IA3 cannot be used at the same time.")
            sys.exit(1)
        else:
            logger.warning("No PEFT model is used.")
        
        logger.success(f"PEFT configured successfully.")
    except Exception as e:
        logger.error(f"Error configuring PEFT: {e}")
        sys.exit(1)

    # define trainer
    compute_metrics = get_compute_metrics(task_details["type"])
    trainer = transformers.Trainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=get_callbacks(model_args),
            )
    
    # initialize wandb
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        logger.info("Wandb API key found in environment variables.")
        try:
            wandb.login(key=wandb_api_key)
            run = wandb.init(
                project=data_args.project_name,
                name=generate_unique_run_name(model_args.hf_model_path, data_args.task_name)
            )
            logger.success("wandb initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing wandb: {e}")
            sys.exit(1)
    else:
        logger.warning("Wandb API key not found in environment variables. Skipping wandb initialization.")
    
    # finetune
    try:
        logger.debug(f"Start training model on task: {data_args.task_name}")

        trainer.train()
        trainer.save_model(training_args.output_dir)

        logger.success(f"Model trained successfully.")
    except Exception as e:
        logger.error(f"Error training model: {e}")
        sys.exit(1)

    # predict on test set
    pred = trainer.predict(test_dataset)
    logger.info(compute_metrics(pred))

    # save evaluation results
    if training_args.eval_and_save_results:
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, "eval_results.json"), "w") as f:
            json.dump(results, f)
    
    # save prediction values
    if data_args.is_save_predictions:
        pred = trainer.predict(test_dataset)
        save_results(
            pred, 
            task_details["type"], 
            training_args.output_dir, 
            test_dataset["name"]
        )

    # log metrics to wandb
    if wandb_api_key:
        try:
            results = trainer.evaluate(eval_dataset=test_dataset)
            metrics = {**vars(model_args), **vars(data_args), **vars(training_args), **results}
            run.log({"metrics": wandb.Table(dataframe=pd.DataFrame([metrics]))})
            wandb.finish()

            logger.success("wandb run completed successfully.")
        except:
            logger.error("Error logging metrics to wandb.")
            sys.exit(1)


def main():
    """fine-tune Huggingface DNA foundation models."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logger.add(os.path.join(training_args.output_dir, "train_and_evaulate.log"), rotation="10 MB")

    try:
        if data_args.use_nt_kmer:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.hf_model_path)
            logger.info("Load Nucleotide Transformer kmer tokenizer")
        else:
            tokenizer =  KmerTokenizer(
                    vocab_file=f"kmer/kmer_{data_args.kmer_window}/vocab.txt",
                    max_length=512,
                    model_max_length=512,
                    k=data_args.kmer_window,
                    stride=data_args.kmer_stride,
                )
            logger.info("Load reconstruct DNABERT kmer tokenizer")
        logger.success(f"Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        sys.exit(1) 
    
    task_details = pgb_dataset_details(data_args)
    if data_args.do_all_tasks:
        # train on all tasks
        logger.info("Train on all tasks.")
        for detail_name in task_details["tasks"]:
            data_args.task_name = data_args.task_name + "." + detail_name
            logger.info(f"Current task name: {data_args.task_name}")
            train(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                tokenizer=tokenizer,
                task_details=task_details
            )
            data_args.task_name = data_args.task_name.split(".")[0] # reset task_name
    else:
        # train on single task
        logger.info("Train on single task.")
        train(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            tokenizer=tokenizer,
            task_details=task_details
        )


if __name__ == "__main__":
    main()
