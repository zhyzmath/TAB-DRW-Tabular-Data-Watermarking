import math
import random
import hashlib
import numpy as np
import pandas as pd
from watermark_tool.generate_original_data import *
from watermark_tool.tool import *
from watermark_tool.watermark import *

def custom_rank(X):
    ranked = np.zeros_like(X, dtype=int)
    
    for col in range(X.shape[1]):
        column_data = X[:, col]
        sorted_indices = np.argsort(column_data)
        sorted_data = column_data[sorted_indices]
        
        current_rank = 0
        ranked_values = np.zeros_like(sorted_data, dtype=int)
        i = 0
        
        while i < len(sorted_data):
            current_value = sorted_data[i]
            count = 1
            while i + count < len(sorted_data) and sorted_data[i + count] == current_value:
                count += 1
            
            ranked_values[i:i+count] = current_rank
            
            current_rank += count
            i += count
        
        ranked_column = np.empty_like(ranked_values)
        ranked_column[sorted_indices] = ranked_values
        ranked[:, col] = ranked_column
    
    return ranked

def get_seed(center, i=0, secret="4832463629493"):
    raw = f"{center:.6f}_{i:.6f}_{secret}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    seed = int(h, 16) & 0xffffffff 
    return seed

def sample_01_from_seed(seed):
    seed = int(seed)
    random.seed(seed)
    return random.getrandbits(1)

def generate_binary_mapping(b, seed):
    random.seed(seed)
    bins = list(range(b))
    random.shuffle(bins)
    
    half = b // 2
    
    mapping = [None] * b
    for i, bin_val in enumerate(bins):
        mapping[bin_val] = 0 if i < half else 1
    return mapping

def transform_x_bin_num(x_bin_num, out_seed, b):
    n, m = x_bin_num.shape
    result = np.empty_like(x_bin_num)
    
    for col in range(m):
        seed = int(get_seed(col+out_seed))
        mapping = generate_binary_mapping(b, seed=seed)
        for row in range(n):
            bin_val = x_bin_num[row, col]
            result[row, col] = mapping[bin_val]
    
    return result

def set_psuedo_bit_pair(x, i=0, b=4):
    # x_rank = custom_rank(x)
    x_rank = np.argsort(np.argsort(x, axis=0), axis=0)
    x_rank_score = x_rank / (x_rank.shape[0] - 1)
    x_bin_num = x_rank_score * b
    x_bin_num = x_bin_num.astype(int)
    x_bin_num[x_bin_num == b] = b - 1
    
    S = transform_x_bin_num(x_bin_num, i, b)
    # S_1, _ = generate_data_TP(S.shape[0], p=S.shape[1])
    row_corr = np.corrcoef(S)
    col_corr = np.corrcoef(S, rowvar=False)
    return S

def set_psuedo_bit(x, N, p, out_seed=0, num_col_idx=None, cat_col_idx=None, num_compute=3):
    out_seed = 0
    target = (p-1) // 2
    p_up = math.ceil(target / 2)
    p = ((p-1) // 2) // 2
    
    S = []
    
    x_score = YJT(x)[0]
    
    seed = int(get_seed(out_seed))
    np.random.seed(seed)
    num_col_idx_selected = np.random.choice(num_col_idx[1:], 
                                            size=len(num_col_idx) // 2 if len(num_col_idx) // 2 > 0 else 1, 
                                            replace=False).tolist()
    cat_col_idx_selected = np.random.choice(cat_col_idx, 
                                            size=len(cat_col_idx) // 2 if len(cat_col_idx) // 2 > 0 else 1, 
                                            replace=False).tolist()
    
    candidate_index = sorted([num_col_idx[0]] + num_col_idx_selected + cat_col_idx_selected)
    x_score = YJT(x[:, candidate_index])[0]
    x_score_sum = np.sum(x_score, axis=1)
    
    x_score_sum_rank = np.argsort(np.argsort(x_score_sum))
    x_score_sum_rank = x_score_sum_rank / (len(x_score_sum_rank) - 1)
    
    x_in = x_score_sum_rank
    
    np.random.seed(207)
    shift = np.random.randint(1, 2**p_up + 1)

    for i in range(N):
        S_one_row = []
        for j in range(1, p+1):
            x = x_in[i] + shift / (2**p_up)
            x = x if x <= 1 else x-1
            index = pow(2, j) * x
            if int(index) == index:
                if index !=0:
                    index = index - 1
            index = int(index) % 4
            if index == 0 or index == 3:
                S_one_row.extend([1, 0])
            else: 
                S_one_row.extend([0, 1])
        if p_up != p:
            x = x_in[i] + shift / (2**p_up)
            x = x if x <= 1 else x-1
            index = pow(2, p_up) * x
            if int(index) == index:
                if index !=0:
                    index = index - 1
            index = int(index) % 4
            if index == 0 or index == 3:
                S_one_row.extend([1])
            else: 
                S_one_row.extend([0])
        S_one_row = np.array(S_one_row)
        S.append(S_one_row)
    S = np.array(S)
    assert S.shape == (N, p*2+(p_up-p))
    return S
