import numpy as np
from scipy.fft import fft, ifft
from watermark_tool.tool import *
import hashlib

def get_seed(center, i=0, secret="4832463629493"):
    raw = f"{center:.6f}_{i:.6f}_{secret}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    seed = int(h, 16) & 0xffffffff
    return seed

def watermark_TABDRW(x, S, k, gamma, delta, discrete=False):
    x_scaled, params = YJT(x)
    y = np.fft.fft(x_scaled, axis=1)
    y_wm = np.copy(y)
    num=0
    for row in range(y.shape[0]):
        y_row = y[row, :].copy()
        y_subset = y_row[1:S.shape[1] + 1]
        abs_imag = np.abs(np.imag(y_subset))
        ranks = np.argsort(np.argsort(abs_imag))
        y_score = ranks / (len(ranks) - 1) if len(ranks) > 1 else ranks

        assert len(y_score) == S.shape[1]
        for i in range(1, S.shape[1] + 1):
            condition_1 = (y_row[i].imag > 0 and S[row, i - 1] == 0) or (y_row[i].imag < 0 and S[row, i - 1] == 1)
            condition_2 = gamma > y_score[i - 1]              
            if condition_1 and condition_2:
                num+=1
                y_row[i] = y_row[i].real - delta * y_row[i].imag * 1j
                y_row[-i] = y_row[-i].real - delta * y_row[-i].imag * 1j
        y_wm[row, :] = y_row
    x_wm_scaled = np.fft.ifft(y_wm, axis=1).real
    x_wm = IYJT(x_wm_scaled, params)
    
    if type(discrete) == np.ndarray:
        x_wm[:, discrete] = np.round(x_wm[:, discrete], decimals=0)
    for col in range(x_wm.shape[1]):
        x_wm[:, col] = np.clip(x_wm[:, col], np.min(x[:, col]), np.max(x[:, col]))
    print(num)
    return x_wm

def watermark_data_RGL(X, green_list, discrete=False):
    if type(discrete) == np.ndarray:
        X[:, discrete] = np.round(X[:, discrete], decimals=0)
    
    assert len(green_list) % 2 == 0, "The length of green list must be even."
    m = len(green_list) // 2
    def find_closest_green_interval(fractional_part):
        index = int(fractional_part * 2 * m)
        frac = fractional_part * 2 * m - index
        shift = 1 if frac >= 0.5 else -1
        if index + shift < 0 or index + shift >= 2 * m:
            shift = -shift
        if green_list[index] != 1:
            if green_list[index+shift] == 1:
                fractional_part = np.random.uniform((index+shift) / (2 * m), (index+shift + 1) / (2 * m))
            else:
                fractional_part = np.random.uniform((index-shift) / (2 * m), (index-shift + 1) / (2 * m))
        return fractional_part

    def wm(x):
        if abs(x) < 1:
            return x
        else:
            int_part = int(x)
            fractional_part = x - int_part
            if fractional_part == 0:
                x = x / 10.0
                int_part = int(x)
                fractional_part = x - int_part
                if fractional_part == 0:
                    return x * 10.0
                sign = -1 if fractional_part < 0 else 1
                modified_fractional_part = find_closest_green_interval(abs(fractional_part))
                x = int_part + sign * modified_fractional_part
                x = x * 10.0
            else:
                sign = -1 if fractional_part < 0 else 1
                modified_fractional_part = find_closest_green_interval(abs(fractional_part))
                x = int_part + sign * modified_fractional_part
            return x
        
    func = np.vectorize(wm)
    x_wm = func(X)
    if type(discrete) == np.ndarray:
        x_wm[:, discrete] = np.round(x_wm[:, discrete], decimals=0)
    for col in range(x_wm.shape[1]):
        x_wm[:, col] = np.clip(x_wm[:, col], np.min(X[:, col]), np.max(X[:, col]))
    return x_wm

def watermark_data_tabmark(X, wm_col_index, p=25, k=500, n_w=0.1, out_seed=0):
    n_rows = X.shape[0]     
    
    X_wm = X.copy()

    domain_edges = np.linspace(-p, p, k + 1)  
    domain_ids   = np.arange(k)             
    
    np.random.seed(out_seed)
    key_rows = np.sort(np.random.choice(n_rows, size=int(n_w*n_rows), replace=False))
    for row in key_rows:
        key_cell = X[row, wm_col_index].copy()
        local_seed = get_seed(int(key_cell), out_seed)

        shuffled = domain_ids.copy()
        np.random.seed(local_seed)
        np.random.shuffle(shuffled)
        green_ids = shuffled[: k // 2]
        d = np.random.choice(green_ids)
        low, high = domain_edges[d], domain_edges[d + 1]
        r = np.random.uniform(low, high)

        X_wm[row, wm_col_index] += r
    return X_wm

def watermark_data_muse(X, X_real):
    X = np.asarray(X)
    X_real = np.asarray(X_real)
    assert X.ndim == 2 and X_real.ndim == 2, "X and X_real must be 2D arrays"
    assert X.shape[1] == X_real.shape[1], "X and X_real must have same number of columns (features)"
    assert X.shape[0] % 2 == 0, "The number of rows in X must be even"
    
    n_rows, n_cols = X.shape
    half = n_rows // 2
    if half == 0:
        return X.copy()

    X_wm = np.empty((half, n_cols), dtype=X.dtype)

    real_sorted_cols = [np.sort(X_real[:, j]) for j in range(n_cols)]
    real_counts = np.array([len(col) for col in real_sorted_cols], dtype=float)
    
    for i in range(half):
        a = X[i]
        b = X[i + half]

        p_a = percentile_vector(a, real_sorted_cols, real_counts)
        p_b = percentile_vector(b, real_sorted_cols, real_counts)

        key_a = select_groups_indices(p_a, 1, n_cols)
        key_b = select_groups_indices(p_b, 1, n_cols)

        rng_a = np.random.RandomState(2333+key_a)
        rng_b = np.random.RandomState(2333+key_b)
        score_a = rng_a.randint(0, 2)
        score_b = rng_b.randint(0, 2)

        winner = a if score_a >= score_b else b
        X_wm[i] = winner
    return X_wm