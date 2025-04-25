from snorkel.labeling import labeling_function

# 常量定义
SELETED = 1
FILTERED = 0
ABSTAIN = -1

# 阈值定义
SCORE_DICT = {
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
    # 以下是额外需要的键
    "grayscale_score": 0,
    "odd_size_score": 0,
    "avg_ratio": 0,
    "num_detections": 0,
    "avg_score": 0,
    "max_score": 0
}

# 从 SCORE_DICT 生成 OPS_LIST
OPS_LIST = list(SCORE_DICT.keys())

import numpy as np

@labeling_function()
def check_blurry_score(x):
    id = OPS_LIST.index("blurry_score")
    if x[id] >= SCORE_DICT["blurry_score"]+0.05:return ABSTAIN
    elif x[id] <= SCORE_DICT["blurry_score"]-0.01:return FILTERED
    return ABSTAIN

@labeling_function()
def check_dark_score(x):
    id = OPS_LIST.index("dark_score")
    if x[id] >= SCORE_DICT["dark_score"]+0.05:return ABSTAIN
    elif x[id] <= SCORE_DICT["dark_score"]-0.05:return FILTERED
    return ABSTAIN

@labeling_function()
def check_grayscale_score(x):
    id = OPS_LIST.index("grayscale_score")
    return ABSTAIN if x[id] == SCORE_DICT["grayscale_score"] else FILTERED

@labeling_function()
def check_light_score(x):
    id = OPS_LIST.index("light_score")
    if x[id] >= SCORE_DICT["light_score"]+0.05:return ABSTAIN
    elif x[id] <= SCORE_DICT["light_score"]-0.05:return FILTERED
    return ABSTAIN

@labeling_function()
def check_low_information_score(x):
    id = OPS_LIST.index("low_information_score")
    if float(x[id]) >= float(SCORE_DICT["low_information_score"])+0.05:return ABSTAIN
    elif float(x[id]) <= float(SCORE_DICT["low_information_score"])-0.05:return FILTERED
    return ABSTAIN

@labeling_function()
def check_odd_aspect_ratio_score(x):
    id = OPS_LIST.index("odd_aspect_ratio_score")
    if float(x[id]) >= float(SCORE_DICT["odd_aspect_ratio_score"])+0.0:return ABSTAIN
    elif float(x[id]) <= float(SCORE_DICT["odd_aspect_ratio_score"])-0.0:return FILTERED
    return ABSTAIN

@labeling_function()
def check_odd_size_score(x):
    id = OPS_LIST.index("odd_size_score")
    if float(x[id]) >= float(SCORE_DICT["odd_size_score"])+0.05:return ABSTAIN
    elif float(x[id]) <= float(SCORE_DICT["odd_size_score"])-0.05:return FILTERED
    return ABSTAIN

@labeling_function()
def check_image_text_similarity(x):
    id = OPS_LIST.index("image_text_similarity")
    if float(x[id]) >= float(SCORE_DICT["image_text_similarity"])+0.02:return SELETED
    elif float(x[id]) <= float(SCORE_DICT["image_text_similarity"])-0.02:return FILTERED
    return ABSTAIN

@labeling_function()
def check_char_rep_ratio(x):
    id = OPS_LIST.index("char_rep_ratio")
    if float(x[id]) >= float(SCORE_DICT["char_rep_ratio"])+0.02:return FILTERED
    elif float(x[id]) <= float(SCORE_DICT["char_rep_ratio"])-0.02:return ABSTAIN
    return ABSTAIN

@labeling_function()
def check_lang_and_lang_score(x):
    id1 = OPS_LIST.index("lang")
    id2 = OPS_LIST.index("lang_score")
    return ABSTAIN if x[id1] == SCORE_DICT["lang"] and float(x[id2])>float(SCORE_DICT["lang_score"]) else FILTERED

@labeling_function()
def check_word_rep_ratio(x):
    id = OPS_LIST.index("word_rep_ratio")
    if float(x[id]) >= float(SCORE_DICT["word_rep_ratio"])+0.05:return FILTERED
    elif float(x[id]) <= float(SCORE_DICT["word_rep_ratio"])-0.05:return ABSTAIN
    return ABSTAIN

@labeling_function()
def check_hclip(x):
    id = OPS_LIST.index("hclip_image_text_similarity")
    if float(x[id]) >= float(SCORE_DICT["hclip_image_text_similarity"])+0.02:return SELETED
    elif float(x[id]) <= float(SCORE_DICT["hclip_image_text_similarity"])-0.02:return FILTERED
    return ABSTAIN

@labeling_function() 
def check_vclip(x):
    id = OPS_LIST.index("vclip_image_text_similarity")
    if float(x[id]) >= float(SCORE_DICT["vclip_image_text_similarity"])+0.02:return SELETED
    elif float(x[id]) <= float(SCORE_DICT["vclip_image_text_similarity"])-0.02:return FILTERED
    return ABSTAIN

@labeling_function()
def check_avg_ratio(x):
    id = OPS_LIST.index("avg_ratio")
    if float(x[id]) > 0.05 and float(x[id]) < float(SCORE_DICT["avg_ratio"]) :return ABSTAIN
    else: return FILTERED

@labeling_function()
def check_num_detections(x):
    id = OPS_LIST.index("num_detections")
    if float(x[id]) >= 1 and float(x[id]) <= float(SCORE_DICT["num_detections"]) :return SELETED
    if float(x[id]) > float(SCORE_DICT["num_detections"]): return ABSTAIN
    else: return FILTERED

@labeling_function()
def check_avg_score(x):
    id = OPS_LIST.index("avg_score")
    if float(x[id]) >= float(SCORE_DICT["avg_score"]) + 0.01 :return SELETED 
    elif float(x[id]) <= float(SCORE_DICT["avg_score"]) - 0.01 : return FILTERED
    else: return ABSTAIN

@labeling_function()
def check_max_score(x):
    id = OPS_LIST.index("max_score")
    if float(x[id]) >= float(SCORE_DICT["max_score"]) + 0.2:return SELETED
    elif float(x[id]) <= float(SCORE_DICT["max_score"]) - 0.2: return FILTERED
    else: return ABSTAIN

@labeling_function()
def check_icc_score(x):
    id = OPS_LIST.index("icc_score")
    if float(x[id]) >= float(SCORE_DICT["icc_score"]) + 0.00 :return SELETED
    elif float(x[id]) <= float(SCORE_DICT["icc_score"]) - 0.03 : return FILTERED
    else: return ABSTAIN

@labeling_function()
def check_gdino_v1(x):
    id = OPS_LIST.index("scores")
    count = sum(1 for item in x[id] if float(item) > 0.3)
    if count >= 5: 
        return SELETED
    elif count >= 1: 
        return ABSTAIN
    else: 
        return FILTERED 

@labeling_function()
def check_gdino_v2(x):
    indices=[]
    id_scores = OPS_LIST.index("scores")
    id_bbox = OPS_LIST.index("boxes")
    indices = [index for index, value in enumerate(x[id_scores]) if float(value) > 0.3]

    if len(indices)!=0:
        selected_bbox = [x[id_bbox][index] for index in indices]

        aspect_ratios = []
        for bbox in selected_bbox:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = min(width, height)/max(width, height)
            aspect_ratios.append(aspect_ratio)

        average_aspect_ratio = np.mean(aspect_ratios)

        if average_aspect_ratio>=0.05 and average_aspect_ratio<=0.95: return ABSTAIN

    return FILTERED 

@labeling_function()
def check_5llv(x):
    id_blurry = OPS_LIST.index("blurry_score")
    id_dark = OPS_LIST.index("dark_score")
    id_light =OPS_LIST.index("light_score")
    id_lowinfo = OPS_LIST.index("low_information_score")
    id_ratio =OPS_LIST.index("odd_aspect_ratio_score")

    if x[id_blurry]>=0.29 and x[id_dark]>=0.32 and x[id_light]>=0.05 and x[id_lowinfo]>=0.3 and x[id_ratio]>=0.33:
        return ABSTAIN
    else:
        return FILTERED 