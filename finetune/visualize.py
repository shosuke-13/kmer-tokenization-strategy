import os
from typing import Tuple, List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde
from sklearn.metrics import pairwise_distances
from loguru import logger

def plot_averages(preds: Tuple[np.ndarray, np.ndarray], output_dir: str) -> None:
    """
    Plot prediction-observed graph for all tissues.
    
    Args:
        preds (Tuple[np.ndarray, np.ndarray]): Tuple of predictions and true labels.
        output_dir (str): Directory to save the output plot.
    """
    pred_all = preds.predictions.squeeze().flatten()
    true_all = preds.label_ids.squeeze().flatten()

    r2 = r2_score(true_all, pred_all)
    logger.info(f"R2 Score (average of all tissues): {r2:.2f}")

    fig, ax = plt.subplots(figsize=(10, 10))
    # fig.suptitle("Prediction-Actual Plot for All Tissues", fontsize=24, y=0.95)

    ax.set_facecolor('#E6E6FA')

    xy = np.vstack([pred_all, true_all])
    kde = gaussian_kde(xy)

    ax_max = max(pred_all.max(), true_all.max())
    xx, yy = np.mgrid[0:ax_max:200j, 0:ax_max:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    z = np.reshape(kde(positions).T, xx.shape)
    cf = ax.contourf(xx, yy, z, levels=20, cmap='Purples')

    ax.set_xlabel("Prediction", fontsize=14)
    ax.set_ylabel("Actual", fontsize=14)
    ax.text(0.05, 0.95, f"R2 = {r2:.2f}", transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    ax.set_xlim(0, ax_max)
    ax.set_ylim(0, ax_max)
    ax.plot([0, ax_max], [0, ax_max], 'b--', alpha=0.5, linewidth=0.5)
    ax.set_aspect('equal')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor('black')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_output = os.path.join(output_dir, "prediction_actual_plot_all_tissues.png")
    plt.savefig(plot_output, dpi=300, bbox_inches='tight')
    logger.info(f"Prediction-Actual Plot for All Tissues saved to {plot_output}")
    plt.close(fig)

def plot_tissue_specific(
        preds: Tuple[np.ndarray, np.ndarray], 
        output_dir: str, 
        num_rows: int, 
        num_cols: int,
        tissues: List[str]
    ) -> None:
    """
    Plot prediction-Actual graphs for each tissue.
    
    Args:
        preds (Tuple[np.ndarray, np.ndarray]): Tuple of predictions and true labels.
        output_dir (str): Directory to save the output plot.
        num_rows (int): Number of rows in the subplot grid.
        num_cols (int): Number of columns in the subplot grid.
        tissues (List[str]): List of tissue names.
    """
    pred_labels = preds.predictions.squeeze()
    true_labels = preds.label_ids.squeeze()

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(40, 30))
    # fig.suptitle("Prediction-Actual Plots for All Tissues", fontsize=24, y=0.95)

    for i, tissue in enumerate(sorted(tissues)):
        pred = pred_labels[:, i]
        true = true_labels[:, i]
        r2 = r2_score(true, pred)

        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        ax.set_facecolor('#E6E6FA')

        xy = np.vstack([pred, true])
        kde = gaussian_kde(xy)

        ax_max = max(pred.max(), true.max())
        xx, yy = np.mgrid[0:ax_max:100j, 0:ax_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        z = np.reshape(kde(positions).T, xx.shape)

        cf = ax.contourf(xx, yy, z, levels=20, cmap='Purples')
        ax.set_title(tissue, fontsize=14)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Actual")
        ax.text(0.05, 0.95, f"R2 = {r2:.2f}", transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        ax.set_xlim(0, ax_max)
        ax.set_ylim(0, ax_max)
        ax.plot([0, ax_max], [0, ax_max], 'b--', alpha=0.5, linewidth=0.5)
        ax.set_aspect('equal')

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor('black')

    for i in range(len(tissues), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_output = os.path.join(output_dir, "prediction_actual_plots_tissues.png")
    plt.savefig(plot_output, dpi=300, bbox_inches='tight')
    logger.info(f"Prediction-Actual Plots for Tissues saved to {plot_output}")
    plt.close(fig)

def plot_expression_profiles(
        preds: Tuple[np.ndarray, np.ndarray],
        tissues: List[str], 
        output_dir: str, 
        mode: str = 'side_by_side'
    ) -> None:
    """
    Create heatmaps for tissue expression profiles.
    
    Args:
        preds (Tuple[np.ndarray, np.ndarray]): Tuple of predictions and true labels.
        tissues (List[str]): List of tissue names.
        output_dir (str): Directory to save the output plot.
        mode (str): 'side_by_side' or 'separate' for plot layout.
    """
    pred_labels = preds.predictions.squeeze()
    true_labels = preds.label_ids.squeeze()

    pred_exp_df = pd.DataFrame(pred_labels, columns=sorted(tissues))
    true_exp_df = pd.DataFrame(true_labels, columns=sorted(tissues))

    pred_exp_df = pred_exp_df[tissues]
    true_exp_df = true_exp_df[tissues]

    true_distance_matrix = pd.DataFrame(pairwise_distances(true_exp_df.T, metric='euclidean'),
                                        index=true_exp_df.columns, columns=true_exp_df.columns)
    pred_distance_matrix = pd.DataFrame(pairwise_distances(pred_exp_df.T, metric='euclidean'),
                                        index=pred_exp_df.columns, columns=pred_exp_df.columns)

    np.fill_diagonal(true_distance_matrix.values, 0)
    np.fill_diagonal(pred_distance_matrix.values, 0)
    
    for color_map in ['Blues', 'Purples', 'Greens', 'Reds']:
        if mode == 'side_by_side':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 16), sharex=True, sharey=True)
            sns.set(font_scale=1.2)

            sns.heatmap(true_distance_matrix, cmap=color_map, linewidths=0.5, linecolor="white", square=True,
                        vmin=0, vmax=np.max(true_distance_matrix.values), annot=False, cbar=True, ax=ax1,
                        cbar_kws={"orientation": "vertical", "shrink": 0.8, "pad": 0.05, "label": "True Euclidean Distance"})
            # ax1.set_title("True Expression Profiles", fontsize=16)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, fontsize=12)
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=12)

            sns.heatmap(pred_distance_matrix, cmap=color_map, linewidths=0.5, linecolor="white", square=True,
                        vmin=0, vmax=np.max(pred_distance_matrix.values), annot=False, cbar=True, ax=ax2,
                        cbar_kws={"orientation": "vertical", "shrink": 0.8, "pad": 0.05, "label": "Predicted Euclidean Distance"})
            ax2.set_title("Predicted Expression Profiles", fontsize=16)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontsize=12)
            ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=12)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            heatmap_output = os.path.join(output_dir, "tissue_expression_profiles_comparison.png")
            plt.savefig(heatmap_output, dpi=300)
            logger.info(f"Tissue Expression Profiles Comparison saved to {heatmap_output}")
            plt.close(fig)

        elif mode == 'separate':
            for data, title, filename in zip([true_distance_matrix, pred_distance_matrix],
                                            ["True Expression Profiles", "Predicted Expression Profiles"],
                                            ["true_expression_profiles.png", "predicted_expression_profiles.png"]):
                fig, ax = plt.subplots(figsize=(20, 16))
                sns.set(font_scale=1.2)

                sns.heatmap(data, cmap=color_map, linewidths=0.5, linecolor="white", square=True,
                            vmin=0, vmax=np.max(data.values), annot=False, cbar=True, ax=ax,
                            cbar_kws={"orientation": "vertical", "shrink": 0.8, "pad": 0.05, "label": "Euclidean Distance"})
                # ax.set_title(title, fontsize=16)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

                plt.tight_layout()

                heatmap_output = os.path.join(output_dir, filename)
                plt.savefig(heatmap_output, dpi=300)
                logger.info(f"{title} saved to {heatmap_output}")
                plt.close(fig)
