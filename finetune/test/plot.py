import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from collections import namedtuple

import numpy as np
from datasets import load_dataset
from visualize import plot_averages, plot_tissue_specific, plot_expression_profiles

def load_and_prepare_data():
    dataset = load_dataset(
                    "InstaDeepAI/plant-genomic-benchmark", 
                    task_name="gene_exp.arabidopsis_thaliana", 
                    trust_remote_code=True
                )
    # test dataset
    test_data = dataset['test']
    true_labels = np.array(test_data['labels'])
    
    # psuedo predictions
    predictions = true_labels + np.random.normal(0, 0.1, true_labels.shape)
    PredictionOutput = namedtuple('PredictionOutput', ['predictions', 'label_ids'])
    preds = PredictionOutput(predictions=predictions, label_ids=true_labels)
    
    # get tissue names
    with open("../expression_tissues", 'r') as file:
            config = yaml.safe_load(file)
            tissues = config["arabidopsis_thaliana"]
    
    return preds, tissues

def run_visualization_test():
    preds, tissues = load_and_prepare_data()
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # visualize functions
    plot_averages(preds, output_dir)
    plot_tissue_specific(preds, output_dir, 7, 8, tissues)
    plot_expression_profiles(preds, tissues, output_dir, mode='side_by_side')
    plot_expression_profiles(preds, tissues, output_dir, mode='separate')

if __name__ == "__main__":
    run_visualization_test()