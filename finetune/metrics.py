from typing import Dict
import pandas as pd
import numpy as np
from finetune.dataset import Dataset
from sklearn.metrics import r2_score
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    matthews_corrcoef, 
    roc_auc_score, 
    auc,
    precision_recall_curve, 
    mean_squared_error, 
    mean_absolute_error
)


def calculate_r2_scores_by_experiment(
        pred, 
        eval_dataset: Dataset
    ) -> Dict[str, float]:
    """Calculate R2 scores for each experiment on promoter/terminator strength"""
    results_df = pd.DataFrame({
                    "name": eval_dataset["name"],
                    "true_label": pred.label_ids,
                    "pred_label": pred.predictions.squeeze()
                })

    # Extract unique suffixes
    suffixes = results_df['name'].apply(lambda x: x.split('_')[-1]).unique()

    r2_scores = {}
    for suffix in suffixes:
        filtered_df = results_df[results_df['name'].apply(lambda x: x.split('_')[-1] == suffix)]
        if filtered_df.empty:
            r2_scores[suffix] = None
        else:
            r2_scores[suffix] = r2_score(filtered_df["true_label"], filtered_df["pred_label"])
    
    return r2_scores


def compute_metrics_for_classification(pred):
    logits, labels = pred
    preds = logits.argmax(-1)
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


def compute_metrics_for_single_label_regression(pred):
    logits, labels = pred
    preds = logits.squeeze()
    return {
        "mse": mean_squared_error(labels, preds),
        "rmse": np.sqrt(mean_squared_error(labels, preds)),
        "mae": mean_absolute_error(labels, preds),
        "r2": r2_score(labels, preds)
    }


def compute_metrics_for_multi_label_regression(pred):
    logits, labels = pred
    preds = logits
    metrics = {
        "mse": mean_squared_error(labels, preds),
        "rmse": np.sqrt(mean_squared_error(labels, preds)),
        "mae": mean_absolute_error(labels, preds),
        "r2": r2_score(labels, preds)
    }

    for i in range(labels.shape[1]):
        metrics[f"r2_var_{i}"] = r2_score(labels[:, i], preds[:, i])

    return metrics
