from snorkel.labeling import labeling_function

# Constants definition
SELECTED = 1
FILTERED = 0
ABSTAIN = -1

# Threshold definitions
SCORE_THRESHOLDS = {
    "blurry_score": 0.29,
    "dark_score": 0.32,
    "light_score": 0.05,
    "low_information_score": 0.3,
    "odd_aspect_ratio_score": 0.33,
    "image_text_similarity": 0.21,
    "char_rep_ratio": 0,
    "lang": "en",
    "lang_score": 0.0,
    "word_rep_ratio": 0,
    "hclip_image_text_similarity": 0,
    "vclip_image_text_similarity": 0,
    "scores": [0],
    "icc_score": 0.035,
    "boxes": [0],
    "num_detections_1": 5, 
    # Additional required keys
    "grayscale_score": 0,
    "odd_size_score": 0,
    "avg_ratio": 0,
    "num_detections": 0,
    "avg_score": 0,
    "max_score": 0
}

# Generate OPS_LIST from SCORE_THRESHOLDS
OPS_LIST = list(SCORE_THRESHOLDS.keys())

import numpy as np

@labeling_function()
def check_blurry_score(x):
    id = OPS_LIST.index("blurry_score")
    if x[id] >= SCORE_THRESHOLDS["blurry_score"]+0.05:
        return ABSTAIN
    elif x[id] <= SCORE_THRESHOLDS["blurry_score"]-0.01:
        return FILTERED
    return ABSTAIN

@labeling_function()
def check_dark_score(x):
    id = OPS_LIST.index("dark_score")
    if x[id] >= SCORE_THRESHOLDS["dark_score"]+0.05:
        return ABSTAIN
    elif x[id] <= SCORE_THRESHOLDS["dark_score"]-0.05:
        return FILTERED
    return ABSTAIN

@labeling_function()
def check_grayscale_score(x):
    id = OPS_LIST.index("grayscale_score")
    return ABSTAIN if x[id] == SCORE_THRESHOLDS["grayscale_score"] else FILTERED

@labeling_function()
def check_light_score(x):
    id = OPS_LIST.index("light_score")
    if x[id] >= SCORE_THRESHOLDS["light_score"]+0.05:
        return ABSTAIN
    elif x[id] <= SCORE_THRESHOLDS["light_score"]-0.05:
        return FILTERED
    return ABSTAIN

@labeling_function()
def check_low_information_score(x):
    id = OPS_LIST.index("low_information_score")
    if float(x[id]) >= float(SCORE_THRESHOLDS["low_information_score"])+0.05:
        return ABSTAIN
    elif float(x[id]) <= float(SCORE_THRESHOLDS["low_information_score"])-0.05:
        return FILTERED
    return ABSTAIN

@labeling_function()
def check_odd_aspect_ratio_score(x):
    id = OPS_LIST.index("odd_aspect_ratio_score")
    if float(x[id]) >= float(SCORE_THRESHOLDS["odd_aspect_ratio_score"])+0.0:
        return ABSTAIN
    elif float(x[id]) <= float(SCORE_THRESHOLDS["odd_aspect_ratio_score"])-0.0:
        return FILTERED
    return ABSTAIN

@labeling_function()
def check_odd_size_score(x):
    id = OPS_LIST.index("odd_size_score")
    if float(x[id]) >= float(SCORE_THRESHOLDS["odd_size_score"])+0.05:
        return ABSTAIN
    elif float(x[id]) <= float(SCORE_THRESHOLDS["odd_size_score"])-0.05:
        return FILTERED
    return ABSTAIN

@labeling_function()
def check_image_text_similarity(x):
    id = OPS_LIST.index("image_text_similarity")
    if float(x[id]) >= float(SCORE_THRESHOLDS["image_text_similarity"])+0.02:
        return SELECTED
    elif float(x[id]) <= float(SCORE_THRESHOLDS["image_text_similarity"])-0.02:
        return FILTERED
    return ABSTAIN

@labeling_function()
def check_char_rep_ratio(x):
    id = OPS_LIST.index("char_rep_ratio")
    if float(x[id]) >= float(SCORE_THRESHOLDS["char_rep_ratio"])+0.02:
        return FILTERED
    elif float(x[id]) <= float(SCORE_THRESHOLDS["char_rep_ratio"])-0.02:
        return ABSTAIN
    return ABSTAIN

@labeling_function()
def check_lang_and_lang_score(x):
    lang_id = OPS_LIST.index("lang")
    score_id = OPS_LIST.index("lang_score")
    return ABSTAIN if x[lang_id] == SCORE_THRESHOLDS["lang"] and float(x[score_id]) > float(SCORE_THRESHOLDS["lang_score"]) else FILTERED

@labeling_function()
def check_word_rep_ratio(x):
    id = OPS_LIST.index("word_rep_ratio")
    if float(x[id]) >= float(SCORE_THRESHOLDS["word_rep_ratio"])+0.05:
        return FILTERED
    elif float(x[id]) <= float(SCORE_THRESHOLDS["word_rep_ratio"])-0.05:
        return ABSTAIN
    return ABSTAIN

@labeling_function()
def check_hclip(x):
    id = OPS_LIST.index("hclip_image_text_similarity")
    if float(x[id]) >= float(SCORE_THRESHOLDS["hclip_image_text_similarity"])+0.02:
        return SELECTED
    elif float(x[id]) <= float(SCORE_THRESHOLDS["hclip_image_text_similarity"])-0.02:
        return FILTERED
    return ABSTAIN

@labeling_function() 
def check_vclip(x):
    id = OPS_LIST.index("vclip_image_text_similarity")
    if float(x[id]) >= float(SCORE_THRESHOLDS["vclip_image_text_similarity"])+0.02:
        return SELECTED
    elif float(x[id]) <= float(SCORE_THRESHOLDS["vclip_image_text_similarity"])-0.02:
        return FILTERED
    return ABSTAIN

@labeling_function()
def check_avg_ratio(x):
    id = OPS_LIST.index("avg_ratio")
    if float(x[id]) > 0.05 and float(x[id]) < float(SCORE_THRESHOLDS["avg_ratio"]):
        return ABSTAIN
    else:
        return FILTERED

@labeling_function()
def check_num_detections(x):
    id = OPS_LIST.index("num_detections")
    detection_count = float(x[id])
    threshold = float(SCORE_THRESHOLDS["num_detections"])
    
    if detection_count >= 1 and detection_count <= threshold:
        return SELECTED
    if detection_count > threshold:
        return ABSTAIN
    return FILTERED

@labeling_function()
def check_avg_score(x):
    id = OPS_LIST.index("avg_score")
    if float(x[id]) >= float(SCORE_THRESHOLDS["avg_score"]) + 0.01:
        return SELECTED 
    elif float(x[id]) <= float(SCORE_THRESHOLDS["avg_score"]) - 0.01:
        return FILTERED
    return ABSTAIN

@labeling_function()
def check_max_score(x):
    id = OPS_LIST.index("max_score")
    if float(x[id]) >= float(SCORE_THRESHOLDS["max_score"]) + 0.2:
        return SELECTED
    elif float(x[id]) <= float(SCORE_THRESHOLDS["max_score"]) - 0.2:
        return FILTERED
    return ABSTAIN

@labeling_function()
def check_icc_score(x):
    id = OPS_LIST.index("icc_score")
    if float(x[id]) >= float(SCORE_THRESHOLDS["icc_score"]) + 0.00:
        return SELECTED
    elif float(x[id]) <= float(SCORE_THRESHOLDS["icc_score"]) - 0.03:
        return FILTERED
    return ABSTAIN

@labeling_function()
def check_gdino_v1(x):
    id = OPS_LIST.index("scores")
    high_confidence_count = sum(1 for item in x[id] if float(item) > 0.3)
    
    if high_confidence_count >= 5: 
        return SELECTED
    elif high_confidence_count >= 1: 
        return ABSTAIN
    return FILTERED 

@labeling_function()
def check_gdino_v2(x):
    scores_id = OPS_LIST.index("scores")
    bbox_id = OPS_LIST.index("boxes")
    high_confidence_indices = [index for index, value in enumerate(x[scores_id]) if float(value) > 0.3]

    if high_confidence_indices:
        selected_bboxes = [x[bbox_id][index] for index in high_confidence_indices]

        aspect_ratios = []
        for bbox in selected_bboxes:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = min(width, height) / max(width, height)
            aspect_ratios.append(aspect_ratio)

        average_aspect_ratio = np.mean(aspect_ratios)

        if 0.05 <= average_aspect_ratio <= 0.95:
            return ABSTAIN

    return FILTERED 

@labeling_function()
def check_five_quality_metrics(x):
    blurry_id = OPS_LIST.index("blurry_score")
    dark_id = OPS_LIST.index("dark_score")
    light_id = OPS_LIST.index("light_score")
    low_info_id = OPS_LIST.index("low_information_score")
    ratio_id = OPS_LIST.index("odd_aspect_ratio_score")

    quality_thresholds_met = (
        x[blurry_id] >= 0.29 and 
        x[dark_id] >= 0.32 and 
        x[light_id] >= 0.05 and 
        x[low_info_id] >= 0.3 and 
        x[ratio_id] >= 0.33
    )
    
    return ABSTAIN if quality_thresholds_met else FILTERED