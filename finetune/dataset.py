import yaml
import math
from typing import Optional, Dict, Tuple, List, Any

from finetune.dataset import load_dataset


class PlantGenomicBenchmark:
    def __init__(self, 
                 pgb_config="config/pgb_tasks.yaml",
                 expression_config="config/expression_tasks.yaml"):
        self.pgb_config = pgb_config
        self.expression_config = expression_config

        with open(self.pgb_config, 'r') as f:
            self.pgb_config = yaml.safe_load(f)

        # gene expression prediction config
        with open(self.expression_config, 'r') as f:
            self.expression_config = yaml.safe_load(f)
    
    def load_dataset(self, task_name):
        return load_dataset(
            "InstaDeepAI/plant-genomic-benchmark", 
            task_name=task_name, 
            trust_remote_code=True
        )
    
    def tokenize_function(self, tokenizer, sample: Dict[str, Any], max_length: int) -> Dict[str, Any]:
        """Tokenizes single sequence."""
        sequence = sample["sequence"]
        label_key = "label" if "label" in sample else "labels" # for single-label and multi-labels
        encoded = tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": sample[label_key],
        }
    
    def get_max_seq_length(self, task_name):
        return self.pgb_config[task_name]['max_seq_length']
    
    def get_num_tokens(self, max_seq_len, kmer_window, kmer_stride, model_max_length):
        return min(math.ceil((max_seq_len - kmer_window + 1) / kmer_stride), model_max_length)
    
    def tokenize_dataset(self, dataset, tokenizer, num_tokens):
        tokenized_datasets = dataset.map(
            lambda x: self.tokenize_function(tokenizer, x, num_tokens),
            batched=True
        )
        return tokenized_datasets
    
    def split_dataset(self, tokenized_datasets, test_size=0.1, val_split_seed=42):
        if "validation" in tokenized_datasets.keys():
            return (
                tokenized_datasets["train"],
                tokenized_datasets["validation"],
                tokenized_datasets["test"],
            )
        else:
            train_datasets = tokenized_datasets["train"].train_test_split(test_size=test_size, seed=val_split_seed)
            return train_datasets["train"], train_datasets["test"], tokenized_datasets["test"]

    def get_tissue_name(self, specie_name):
        """get tissue names for a given specie and return number of rows and columns for plot expression profiles."""
        tissues = self.expression_config[specie_name]['tissues']
        num_rows = self.expression_config[specie_name]['num_rows']
        num_cols = self.expression_config[specie_name]['num_cols']
        return tissues, num_rows, num_cols
