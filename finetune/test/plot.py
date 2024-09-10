import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from collections import namedtuple

import numpy as np
from loguru import logger
from datasets import load_dataset
from visualize import plot_averages, plot_tissue_specific, plot_expression_profiles

def load_and_prepare_data():
    try:
        dataset = load_dataset(
                        "InstaDeepAI/plant-genomic-benchmark", 
                        task_name="gene_exp.arabidopsis_thaliana", 
                        trust_remote_code=True
                    )
        logger.success("Dataset loaded successfully")
    except Exception as e:
        logger.error(f"An error occured while loading the dataset: {e}")
        sys.exit(1)

    # test dataset
    test_data = dataset['test']
    true_labels = np.array(test_data['labels'])
    
    # psuedo predictions
    predictions = true_labels + np.random.normal(0, 0.1, true_labels.shape)
    PredictionOutput = namedtuple('PredictionOutput', ['predictions', 'label_ids'])
    preds = PredictionOutput(predictions=predictions, label_ids=true_labels)

    return preds

def run_visualization_test():
    species_name = "arabidopsis_thaliana"
    preds = load_and_prepare_data()

    # get tissue names
    try:
        with open("../expression_tissues", 'r') as file:
                config = yaml.safe_load(file)[species_name]

        tissues = config["tissues"]
        num_rows = config["num_rows"]
        num_cols = config["num_cols"]

        logger.success("Tissue names loaded successfully")
    except FileNotFoundError:
        logger.error("Tissue names yaml file is not found")
        sys.exit(1)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # visualize functions
    logger.info("Running visualization functions")
    plot_averages(preds, output_dir)
    plot_tissue_specific(preds, output_dir, tissues, num_rows=num_rows, num_cols=num_cols)
    plot_expression_profiles(preds, tissues, output_dir, mode='side_by_side')
    plot_expression_profiles(preds, tissues, output_dir, mode='separate')

if __name__ == "__main__":
    run_visualization_test()
