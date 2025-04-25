import os
import json
import ujson
import numpy as np
import pandas as pd
from tqdm import tqdm 
import torch
import random
from concurrent.futures import ThreadPoolExecutor

from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter, LabelModel

# 导入从 labeling_functions.py 中移动的函数和常量
from labeling_functions import (
    SELETED, FILTERED, ABSTAIN, OPS_LIST, SCORE_DICT,
    check_blurry_score, check_dark_score, check_grayscale_score, 
    check_light_score, check_low_information_score, check_odd_aspect_ratio_score,
    check_odd_size_score, check_image_text_similarity, check_char_rep_ratio,
    check_lang_and_lang_score, check_word_rep_ratio, check_hclip, check_vclip,
    check_avg_ratio, check_num_detections, check_avg_score, check_max_score,
    check_icc_score, check_gdino_v1, check_gdino_v2, check_5llv
)

# 导入从 utils.py 中移动的功能性函数
from utils import (
    read_jsonl, read_jsonl_i, read_jsonl_vhclip, read_jsonl_g, read_jsonl_icc,
    process_line, copy_selected_data, process_line_withscore, copy_selected_data_withscore,
    set_seed
)

import time

if __name__=="__main__":

    score_df1 = read_jsonl_i("/mnt/share_disk/LIV/datacomp/processed_data/881w_processed/881w_dedup_stats.jsonl")
    score_df2 = read_jsonl_g("/mnt/share_disk/LIV/datacomp/processed_data/881w_processed/881w_dedup_gdino.jsonl")
    score_df3 = read_jsonl_icc("/mnt/share_disk/LIV/datacomp/processed_data/caption_eval/881w_icc_score.jsonl")

    combined_df = pd.concat([score_df1, score_df2, score_df3], axis=1)
    combined_df.columns = list(range(combined_df.shape[1]))
    # record input index
    combined_df['original_index'] = combined_df.index

    print(combined_df)

    # filter by low-level-visual filters
    combined_df_llv = combined_df

    print(combined_df_llv)

    # select lfs 
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

    t1 = time.time()
    # ops computation
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=combined_df_llv)
    t2 = time.time()  

    # for counting ops
    unique, counts = np.unique(L_train, return_counts=True)
    count_dict = dict(zip(unique, counts))
    
    # cnt
    print("Count of 1s:", count_dict.get(1, 0))
    print("Count of 0s:", count_dict.get(0, 0))
    print("Count of -1s:", count_dict.get(-1, 0))

    #cls balance
    class_cnt_ratio = [count_dict.get(0, 0),count_dict.get(1, 0)]
    class_balance = [class_cnt_ratio[1]/sum(class_cnt_ratio), class_cnt_ratio[0]/sum(class_cnt_ratio)]

    # lf anlysis
    analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    print("## LFAnalysis ##\n",LFAnalysis(L=L_train, lfs=lfs).lf_summary())

    t3 = time.time()
    # train label model
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, 
                    n_epochs=200, 
                    lr=0.01, 
                    l2=0.001,
                    log_freq=20, 
                    seed=123, 
                    # class_balance=class_balance
                    )
    # label_model.fit(L_train=L_train, n_epochs=500, log_freq=20, seed=123, class_balance=class_balance)
    t4 = time.time()

    # get weights
    weights = label_model.get_weights()
    
    # print LFs results
    df_analysis = analysis.copy()
    df_analysis['weights'] = weights
    print(df_analysis)

    # 计算 Coverage, Conflicts, Overlaps 的平均值
    attributes_array = analysis.to_numpy()
    coverage_mean = analysis['Coverage'].mean()
    conflicts_mean = analysis['Conflicts'].mean()
    overlaps_mean = analysis['Overlaps'].mean()

    # 计算 Coverage, Conflicts, Overlaps 的加权平均值
    coverage_weighted_mean = (analysis['Coverage'] * weights).sum() / weights.sum()
    conflicts_weighted_mean = (analysis['Conflicts'] * weights).sum() / weights.sum()
    overlaps_weighted_mean = (analysis['Overlaps'] * weights).sum() / weights.sum()

    print(coverage_mean,conflicts_mean,overlaps_mean)
    print(coverage_weighted_mean,conflicts_weighted_mean,overlaps_weighted_mean)

    t5 = time.time()
    # inference
    probs_train = label_model.predict_proba(L=L_train)
    print("## LabelModel ##\n",probs_train)
    t6 = time.time()

    # obtain top
    score_list = [i[1] for i in probs_train]

    print(len(score_list))

    selected_indices = []
    for i in range(0, len(score_list), 4):
        group = score_list[i:i+4]
        best_index = group.index(max(group)) + i
        selected_indices.append(best_index)
        
    selected_indices.sort(key=lambda x: score_list[x], reverse=True)

    half_length = len(selected_indices) // 2
    top_half_indices = selected_indices[:half_length]

    # random.shuffle(indices)
    output_indices = combined_df_llv.iloc[top_half_indices]['original_index'].values

    print("start preparing jsonl file")

    random.shuffle(output_indices)

    copy_selected_data(
        "/mnt/share_disk/LIV/datacomp/processed_data/881w_processed/881w_dedup.jsonl",
        "/mnt/share_disk/LIV/datacomp/processed_data/881w_processed/best_181_ensemblescore.jsonl",
        output_indices
    )

    print(t2-t1,t4-t3,t6-t5)