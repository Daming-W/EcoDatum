import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm 
import random
from concurrent.futures import ThreadPoolExecutor
import time
import argparse
import yaml

from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel

# Import functions and constants from labeling_functions.py
from labeling_functions import (
    SELECTED, FILTERED, ABSTAIN,
    check_blurry_score, check_odd_aspect_ratio_score,
    check_image_text_similarity, check_lang_and_lang_score,
    check_hclip, check_vclip, check_icc_score, check_gdino_v1
)

# Import utility functions from utils.py
from utils import (
    read_jsonl_i, read_jsonl_gdino, read_jsonl_icc,
    copy_selected_data
)

def parse_args():

    # Create Parser, but only accept config file path
    parser = argparse.ArgumentParser(description='EcoDatum Ensemble Computing')
    parser.add_argument('--config', type=str, default='examples/config/demo.yaml', 
                        help='Path to configuration YAML file')
    args_config = parser.parse_args()
    
    # Check if the config file exists
    if not os.path.exists(args_config.config):
        print(f"The Config File Not Exists: {args_config.config}")
        exit(1)
    
    # Read Config File
    with open(args_config.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert Config to Namespace Object
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    return Args(**config)

if __name__ == "__main__":
    args = parse_args()
    
    # Load data from different sources
    score_df1 = read_jsonl_i(args.stats_path)
    score_df2 = read_jsonl_gdino(args.gdino_path)
    score_df3 = read_jsonl_icc(args.icc_path)

    # Combine dataframes
    combined_df = pd.concat([score_df1, score_df2, score_df3], axis=1)
    combined_df.columns = list(range(combined_df.shape[1]))

    # Record original indices
    combined_df['original_index'] = combined_df.index

    # Select labeling functions
    lfs = [
        check_blurry_score,
        check_odd_aspect_ratio_score,
        check_image_text_similarity,
        check_lang_and_lang_score,
        check_hclip,
        check_vclip,
        check_icc_score,
        check_gdino_v1
    ] 

    # Measure time for operations computation
    t1 = time.time()
    # Apply labeling functions
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=combined_df)
    t2 = time.time()  

    # Count label distribution
    unique, counts = np.unique(L_train, return_counts=True)
    count_dict = dict(zip(unique, counts))
    
    # Print counts
    print("Count of 1s:", count_dict.get(1, 0))
    print("Count of 0s:", count_dict.get(0, 0))
    print("Count of -1s:", count_dict.get(-1, 0))

    # Calculate class balance
    class_cnt_ratio = [count_dict.get(0, 0), count_dict.get(1, 0)]
    class_balance = [class_cnt_ratio[1]/sum(class_cnt_ratio), class_cnt_ratio[0]/sum(class_cnt_ratio)]

    # Analyze labeling functions
    analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    print("LFAnalysis Summary: ", LFAnalysis(L=L_train, lfs=lfs).lf_summary())

    # Measure time for model training
    t3 = time.time()
    # Train label model
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(
        L_train=L_train, 
        n_epochs=args.epochs, 
        lr=args.lr, 
        l2=args.l2,
        log_freq=20, 
        seed=args.seed
    )
    t4 = time.time()

    # Get model weights
    weights = label_model.get_weights()
    
    # Print LF results with weights
    df_analysis = analysis.copy()
    df_analysis['weights'] = weights
    print('LFAnalysis Summary with Weights: ', df_analysis)

    # Calculate mean values for Coverage, Conflicts, Overlaps
    attributes_array = analysis.to_numpy()
    coverage_mean = analysis['Coverage'].mean()
    conflicts_mean = analysis['Conflicts'].mean()
    overlaps_mean = analysis['Overlaps'].mean()

    # Calculate weighted mean values
    coverage_weighted_mean = (analysis['Coverage'] * weights).sum() / weights.sum()
    conflicts_weighted_mean = (analysis['Conflicts'] * weights).sum() / weights.sum()
    overlaps_weighted_mean = (analysis['Overlaps'] * weights).sum() / weights.sum()

    print('Coverage Mean: ', coverage_mean, 'Conflicts Mean: ', conflicts_mean, 'Overlaps Mean: ', overlaps_mean)
    print('Coverage Weighted Mean: ', coverage_weighted_mean, 'Conflicts Weighted Mean: ', conflicts_weighted_mean, 'Overlaps Weighted Mean: ', overlaps_weighted_mean)

    # Measure time for inference
    t5 = time.time()
    # Predict probabilities
    probs_train = label_model.predict_proba(L=L_train)
    t6 = time.time()

    # Extract scores
    score_list = [i[1] for i in probs_train]
    print('The Number of Scores: ', len(score_list))

    # Select best indices from each group
    selected_indices = list(range(len(score_list)))
       
    # Sort by score in descending order
    selected_indices.sort(key=lambda x: score_list[x], reverse=True)
    print(selected_indices)
    # Take top half
    top_indices = selected_indices[:args.output_num]

    # Get original indices
    output_indices = combined_df.iloc[top_indices]['original_index'].values

    print("Start preparing jsonl file")

    # Shuffle output indices
    random.shuffle(output_indices)

    # Copy selected data to output file
    copy_selected_data(
        args.input_jsonl,
        args.output_jsonl,
        output_indices
    )

    print("End of Ensemble Computing, the output jsonl file is saved in: ", args.output_jsonl, "with ", len(output_indices), "data.")