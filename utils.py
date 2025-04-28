import os
import json
import ujson
import numpy as np
import pandas as pd
from tqdm import tqdm 
import torch
import random
from concurrent.futures import ThreadPoolExecutor

def read_jsonl(jsonl_path):
    res_array=[]

    with open(jsonl_path,'r') as f:
        for i in tqdm(f):
            score=json.loads(i.strip())["__dj__stats__"]         
            res_array.append([i[0] if type(i) is list else i for i in score.values()])

    return pd.DataFrame(res_array)

def read_jsonl_i(jsonl_path):
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    res_array = [None] * len(lines)

    for idx, line in enumerate(tqdm(lines, desc="Processing", unit=" lines")):
        score = ujson.loads(line.strip())["__dj__stats__"]
        res_array[idx] = [i[0] if isinstance(i, list) else i for i in score.values()]
    
    return pd.DataFrame(res_array)

def read_jsonl_vhclip(jsonl_path):
    res_array=[]

    with open(jsonl_path,'r') as f:
        for i in tqdm(f):
            score=json.loads(i.strip())     
            res_array.append([i[0] if type(i) is list else i for i in score.values()])

    return pd.DataFrame(res_array)

def read_jsonl_gdino(jsonl_path):
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    res_array = []

    for idx, line in enumerate(tqdm(lines, desc="Processing", unit=" lines")):
        score = ujson.loads(line.strip())["__dj__stats__"].get("grounding_dino_detection", None)
        if score["scores"] is None or score["num_detections"] == 0:
            res_array.append([[0.0]])
        else:
            res_array.append([score["scores"]])
    return pd.DataFrame(res_array)

def read_jsonl_icc(jsonl_path):
    res_array=[]

    with open(jsonl_path,'r') as f:
        for i in tqdm(f):
            score=json.loads(i.strip())     
            res_array.append(score["score"])

    return pd.DataFrame(res_array)

def process_line(args):
    line, i, index_set = args
    if i in index_set:
        data = json.loads(line)
        return json.dumps(data)
    return None

def copy_selected_data(jsonlin, jsonlout, index_list):
    index_set = set(index_list)
    with open(jsonlin, 'r') as infile:
        lines = infile.readlines()

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_line, [(line, i, index_set) for i, line in enumerate(lines)]), total=len(lines)))
        
        with open(jsonlout, 'w') as outfile:
            for result in results:
                if result is not None:
                    outfile.write(result + '\n')

def process_line_withscore(args):
    line, i, index_set, search_dict = args
    if i in index_set:
        data = json.loads(line)
        data["score"] = search_dict[i]
        return json.dumps(data)
    return None

def copy_selected_data_withscore(jsonlin, jsonlout, index_list, score_list):
    index_set = set(index_list)
    search_dict = dict(zip(index_list, score_list))

    with open(jsonlin, 'r') as infile:
        lines = infile.readlines()

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_line_withscore, [(line, i, index_set,search_dict) for i, line in enumerate(lines)]), total=len(lines)))
        
        with open(jsonlout, 'w') as outfile:
            for result in results:
                if result is not None:
                    outfile.write(result + '\n')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 