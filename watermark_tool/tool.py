import json
import numpy as np
import hashlib
from scipy.stats import norm, boxcox, entropy
from scipy.special import inv_boxcox
from scipy.optimize import minimize_scalar
from scipy.stats import yeojohnson
from scipy.interpolate import PchipInterpolator
from sklearn.preprocessing import PowerTransformer

# # Yeo-Johnson normalization
def YJT(X):
    X = np.asarray(X, dtype=np.float64)
    n_features = X.shape[1]
    transformed = np.empty_like(X)
    params = []
    
    for col in range(n_features):
        col_data = X[:, col].copy()
        param = {'lambda': None, 'mean': None, 'std': None, 'shift': None}
        
        if np.all(col_data == col_data[0]):
            np.random.seed(88)
            index = np.random.randint(1, len(col_data))
            col_data[index] += 1e-6
        
        transformed_col, lambda_ = yeojohnson(col_data)
        param['lambda'] = lambda_
        
        mu, sigma = np.mean(transformed_col), np.std(transformed_col)
        param['mean'] = mu
        param['std'] = sigma
        
        transformed[:, col] = (transformed_col - mu) / sigma
        params.append(param)
    
    return transformed, params

# # Yeo-Johnson denormalization
def IYJT(X_trans, params):
    X_trans = np.asarray(X_trans, dtype=np.float64)
    restored = np.empty_like(X_trans)
    
    for col in range(X_trans.shape[1]):
        param = params[col]
        lambda_ = param['lambda']
        mu = param['mean']
        sigma = param['std']
        trans_col = X_trans[:, col]
        
        denorm = trans_col * sigma + mu
        
        orig = np.empty_like(denorm)
        positive_mask = denorm >= 0
        negative_mask = ~positive_mask
        
        if lambda_ == 0:
            orig[positive_mask] = np.exp(denorm[positive_mask]) - 1
        else:
            arg = denorm[positive_mask] * lambda_ + 1
            invalid = arg <= 0
            if np.any(invalid):
                safe_min = 1e-12
                arg[invalid] = safe_min
            orig[positive_mask] = np.power(arg, 1 / lambda_) - 1
        
        if lambda_ == 2:
            orig[negative_mask] = 1 - np.exp(-denorm[negative_mask])
        else:
            arg = -(2 - lambda_) * denorm[negative_mask] + 1
            invalid = arg <= 0
            if np.any(invalid):
                safe_min = 1e-12
                arg[invalid] = safe_min
            orig[negative_mask] = 1 - np.power(arg, 1 / (2 - lambda_))
        
        restored[:, col] = orig
    
    return restored

def compute_T_DRW(y, S):
    assert len(y.shape) == 2
    m = S.shape[1]
    y_subset = y[:, 1:m+1]
    
    condition1 = (y_subset.imag > 0) & (S == 1)
    condition2 = (y_subset.imag < 0) & (S == 0)
    condition = condition1 | condition2
    
    T = np.sum(condition, axis=1)
    return T/m

def compute_T_RGL(x: np.ndarray, green_list: np.ndarray, discrete=None) -> np.ndarray:
    if type(discrete) == np.ndarray:
        x[:, discrete] = np.round(x[:, discrete], decimals=0)
    # Ensure green_list is an array for advanced indexing
    green = np.asarray(green_list)
    m = green.size // 2
    N, p = x.shape

    # 1. Integer and fractional parts
    x_int = np.trunc(x).astype(int)
    x_frac = np.abs(x - x_int)

    # 2. Handle exact-integer entries
    zero_mask = (x_frac == 0)
    if zero_mask.any():
        x_temp = x[zero_mask] / 10.0
        x_int_temp = np.trunc(x_temp).astype(int)
        x_frac[zero_mask] = np.abs(x_temp - x_int_temp)

    # 3. Compute bin index into green_list
    #    floor(x_frac * 2m) ensures index in [0, 2m-1]
    x_index = np.floor(x_frac * (2 * m)).astype(int)

    # 4. Map through green_list and compute row-wise sums
    T = green[x_index]         # shape (N, p)
    statistic = T.sum(axis=1) / p

    return statistic

def empirical_cdf(data):
    # Sort the data
    sorted_data = np.sort(data)
    
    # Compute the ECDF values
    ecdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    return sorted_data, ecdf_values

def wasserstein_distance(x, y, k=1):
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    assert len(x_sorted) == len(y_sorted)
    
    differences = np.abs(x_sorted - y_sorted) ** k
    
    distance = (np.sum(differences) / len(x_sorted)) ** (1 / k)
    return distance

def rank_distance(x, y):
    x_ranks = np.argsort(np.argsort(x))
    y_ranks = np.argsort(np.argsort(y))
    diff = np.abs(x_ranks - y_ranks)
    return np.sum(diff)

def mean_var_bias(x, x_wm):
    mean_bias = np.mean(x_wm) - np.mean(x)
    var_bias = np.var(x_wm) - np.var(x)
    return (mean_bias, var_bias)

def compute_detection_rate(T_values, alpha=0.005):
    num_outliers = 0
    threshold = norm.ppf(1 - alpha)
    for T in T_values:
        if T > threshold:
            num_outliers += 1
    return num_outliers / len(T_values)

def calculate_bin_dispersion(X, n_bins=50, metric='entropy'):
    m, n = X.shape
    scores = np.zeros(n)
    
    for col in range(n):
        column_data = X[:, col]
        
        min_val, max_val = np.min(column_data), np.max(column_data)
        bins = np.linspace(min_val, max_val, n_bins + 1)
        hist, _ = np.histogram(column_data, bins=bins)
        hist = hist.astype(float)
        
        hist += 1e-9 
        
        prob = hist / hist.sum()
        
        if metric == 'entropy':
            scores[col] = entropy(prob, base=2)
        elif metric == 'chi2':
            expected = m / n_bins
            chi2 = np.sum((hist - expected)**2 / expected)
            scores[col] = chi2
        elif metric == 'gini':
            gini = 1 - np.sum(prob**2)
            scores[col] = gini
        elif metric == 'dispersion':
            scores[col] = np.std(hist)
        else:
            raise ValueError("Unspported metric!")
    
    return scores

def add_fields_to_info_json(info_path, new_fields):
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    
    for k, v in new_fields.items():
        info[k] = v
    
    with open(info_path, 'w', encoding='utf-8') as f:

        json.dump(info, f, indent=4, ensure_ascii=False, sort_keys=False)
        
def percentile_vector(row_vec, real_sorted_cols, real_counts):
    n_cols = len(row_vec)
    p = np.empty(n_cols, dtype=float)
    for j in range(n_cols):
        col_sorted = real_sorted_cols[j]
        # Use right insertion point, treat equal values as "<=" count
        cnt = np.searchsorted(col_sorted, row_vec[j], side='right')
        p[j] = cnt / real_counts[j]
    return p

def select_groups_indices(percentiles, group_size=1, n_cols=None):
    if n_cols is None:
        n_cols = len(percentiles)
    
    group_size = max(0, min(group_size, n_cols))
    if group_size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    order_low_to_high = np.argsort(percentiles)
    num_cols = len(order_low_to_high)
    idx_low = order_low_to_high[0]
    idx_high = order_low_to_high[-1]
    idx_mid = order_low_to_high[num_cols//2]
    
    s = f"{idx_low}_{idx_mid}_{idx_high}"
    h = hashlib.sha256(s.encode('utf-8')).hexdigest()
    # print(int(h, 16) & 0xffffffff)
    return int(h, 16) & 0xffffffff
