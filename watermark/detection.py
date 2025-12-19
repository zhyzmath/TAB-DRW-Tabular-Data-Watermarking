import os
import torch
import wandb
import numpy as np
import pandas as pd 
import time
from scipy.fft import fft, ifft
from tabsyn.model import MLPDiffusion, DDIMModel, DDIMScheduler
from tabsyn.latent_utils import get_input_generate, get_encoder_latent, get_decoder_latent
from tabsyn.watermark_utils import eval_watermark
from tabsyn.process_syn_dataset import process_data, preprocess_syn
from watermark_tool.tool import *
from watermark_tool.pseudo_gen import set_psuedo_bit, get_seed
from watermark_tool.generate_original_data import generate_RGL

def get_watermark_metric(
        args, dataname, data_path, save_dir, pre_k, info, model, noise_scheduler,
        watermarking_mask, gt_patch, mean=0, latents=None, X_num=None, X_cat=None,
        k=None, mask_col=None
):
    # Preprocess data
    process_data(name=dataname, data_path=data_path, save_dir=save_dir, k=pre_k)
    if X_num is None or X_cat is None:
        X_num, X_cat = preprocess_syn(save_dir, info['task_type'], k=pre_k)

    # Get the latent of the synthetic tabular from the vae encoder
    # syn_latent_encoder = get_encoder_latent(X_num, X_cat, info, args.device)
    syn_latent = get_decoder_latent(X_num, X_cat, info, args.device, aux=latents, mask_col=mask_col)
    mean = mean.to(args.device)

    syn_latent = (syn_latent - mean) / 2
    # syn_latent = (latents - mean) / 2

    # Reverse the noise using DDIM-Inversion
    reversed_noise = noise_scheduler.gen_reverse(
            model.noise_fn, syn_latent, num_inference_steps=args.steps, eta=0.0
        )

    metric = evaluate_watermark_methods(args, reversed_noise, watermarking_mask, gt_patch, k)

    return metric

def eval_GLW(x, k, out_seed=0, discrete=None):
    out_seed = 0
    S = generate_RGL()
    T_row = compute_T_RGL(x, S, discrete)
    if k == 'no-w':
        return T_row
    else:
        return np.mean(T_row)

def eval_muse(x, k, X_real, out_seed=0):
    # Pre-sort real data for each column to facilitate fast percentile calculation
    real_sorted_cols = [np.sort(X_real[:, j]) for j in range(X_real.shape[1])]
    real_counts = np.array([len(col) for col in real_sorted_cols], dtype=float)
    
    T_row = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        p = percentile_vector(x[i], real_sorted_cols, real_counts)
        key = select_groups_indices(p, 1, X_real.shape[1])
        rng = np.random.RandomState(2333+key)
        score = rng.randint(0, 2)
        T_row[i] = score
    
    if k == 'no-w':
        return T_row
    else:
        return np.mean(T_row)
    
def eval_tabmark(x, k_out, x_ori, wm_col_index, p=25, k=500, n_w=0.1, out_seed=0):
    def msb_primary_key(vals, b=5):
        keys = []
        vals = np.abs(vals) 
        for v in vals:
            iv = int(v)
            s = bin(iv)[2:]      
            if len(s) < b:
                s = s.zfill(b)     
            else:
                s = s[:b]       
            keys.append(s)
        return ''.join(keys)

    n_rows = x_ori.shape[0]
    T_row = np.zeros(x.shape[0])

    domain_edges = np.linspace(-p, p, k + 1)  
    domain_ids   = np.arange(k)             
    
    np.random.seed(out_seed)
    key_rows = np.sort(np.random.choice(n_rows, size=int(n_w*n_rows), replace=False))
    target_key_dict = {}
    for row in range(x.shape[0]):
        sus_cell = x[row, wm_col_index].copy()
        np.random.seed()
        selected_col_idx = np.random.choice(np.delete(np.arange(x.shape[1]), wm_col_index), size=5, replace=False)
        selected_col = x[row, selected_col_idx].copy()
        primary_key = msb_primary_key(selected_col)
        key_row = None
        for key_row in key_rows:
            selected_col = x_ori[key_row, selected_col_idx].copy()
            primary_key_ori = msb_primary_key(selected_col)
            if primary_key == primary_key_ori:
                key_cell = x_ori[key_row, wm_col_index].copy()
                diff = sus_cell - key_cell
                if diff > -p and diff < p:
                    target_key_dict[row] = key_row
                    key_row = -1
                    break
        if key_row != -1 or sus_cell - key_cell == 0:
            np.random.seed()
            T_row[row] = np.random.randint(0, 2)
        else:
            diff = sus_cell - key_cell
            local_seed = get_seed(int(key_cell), out_seed)
            shuffled = domain_ids.copy()
            np.random.seed(local_seed)
            np.random.shuffle(shuffled)
            green_ids = shuffled[: k // 2]
            dom_idx = int((diff + p) // (2 * p / k))
            T_row[row] = 1 if dom_idx in green_ids else 0
            
    if k_out == 'no-w':
        return T_row
    else:
        return np.mean(T_row)

def eval_TAB_DRW(x, with_w, k, out_seed=0, num_col_idx=None, cat_col_idx=None, value_col_idx=None):
    out_seed = 0
    if k == 'w':
        S = set_psuedo_bit(x, x.shape[0], x.shape[1], out_seed, num_col_idx, cat_col_idx)
        x_scaled, params = YJT(x)
    else:
        if value_col_idx is not None:
            x_num = x[:, value_col_idx].copy()
        else:
            x_num = x[:, num_col_idx].copy()
        S = set_psuedo_bit(x, x_num.shape[0], x_num.shape[1], out_seed, num_col_idx, cat_col_idx)
        x_scaled, params = YJT(x_num)
        
    # y_wm = fft(x_scaled, axis=1, norm='ortho')
    y_wm = np.fft.fft(x_scaled, axis=1)
    
    T_row = compute_T_DRW(y_wm, S)
        
    if k == 'no-w':
        return T_row
    else:
        return np.mean(T_row)
    
def evaluate_watermark_methods(args, reversed_noise, watermarking_mask, gt_patch, k=None):
    # Gaussian
    if args.with_w == 'GS':
        metric = eval_GS(reversed_noise, k=k, device=args.device)
    elif args.with_w == 'TabWak':
        metric = eval_TabWak(reversed_noise, k=k)
    elif args.with_w == 'TabWak_star':
        metric = eval_TabWak_star(reversed_noise, k=k)
    else:
        # TreeRing
        metric = eval_watermark(reversed_noise, watermarking_mask, gt_patch, args)
    return metric

def eval_TabWak(reversed_noise, bit_dim=4,k=None):
    for i in range(reversed_noise.shape[0]):
        mid = torch.quantile(reversed_noise[i], 0.5)

        for j in range(reversed_noise.shape[1]):
            if reversed_noise[i][j] <= mid:
                reversed_noise[i][j] = 0
            else:
                reversed_noise[i][j] = 1
    bsz, seq_len = reversed_noise.shape
    torch.manual_seed(217)
    permutation = torch.randperm(seq_len)
    inverse_permutation = torch.argsort(permutation)
    reversed_noise = reversed_noise[:, inverse_permutation]
    half_dim = seq_len // 2
    acc_bit_row_list = []
    for row in reversed_noise:
        first_half = row[:half_dim]
        last_half = row[half_dim:]
        correct_row = (first_half == last_half).sum().item()
        acc_bit_row = correct_row / half_dim
        acc_bit_row_list.append(acc_bit_row)
        # wandb.log({f'{k}-acc_bit_row':acc_bit_row})
        
    if k == 'w':
        avg_bit_accuracy = np.mean(np.array(acc_bit_row_list))
        return avg_bit_accuracy
    else:
        return acc_bit_row_list

def eval_TabWak_star(reversed_noise,k=None):
    cnt = 0
    correct = 0
    for i in range(reversed_noise.shape[0]):
        q1 = torch.quantile(reversed_noise[i], 0.25)
        q2 = torch.quantile(reversed_noise[i], 0.5)
        q3 = torch.quantile(reversed_noise[i], 0.75)
        for j in range(reversed_noise.shape[1]):
            if reversed_noise[i][j] <= q1:
                reversed_noise[i][j] = 0
            elif reversed_noise[i][j] >= q3:
                reversed_noise[i][j] = 1
            elif reversed_noise[i][j] > q1 and reversed_noise[i][j] < q2:
                reversed_noise[i][j] = 2
            elif reversed_noise[i][j] >=q2 and reversed_noise[i][j] < q3:
                reversed_noise[i][j] = 3

    bsz, seq_len = reversed_noise.shape
    torch.manual_seed(217)
    permutation = torch.randperm(seq_len)
    inverse_permutation = torch.argsort(permutation)

    reversed_noise = reversed_noise[:, inverse_permutation]
    half_dim = seq_len // 2
    acc_bit_row_list = []
    for row in reversed_noise:
        cnt_row = 0
        correct_row = 0
        first_half = row[:half_dim]
        last_half = row[half_dim:]
        for i in range(half_dim):
            if first_half[i] == 0 or first_half[i] == 1:
                cnt_row += 1
            if first_half[i] == 0 and (last_half[i] == 0 or last_half[i] == 2):
                correct_row += 1
            if first_half[i] == 1 and (last_half[i] == 1 or last_half[i] == 3):
                correct_row += 1

        cnt += cnt_row
        correct += correct_row

        acc_bit_row = correct_row / cnt_row if cnt_row != 0 else 0.5
        acc_bit_row_list.append(acc_bit_row)
        # wandb.log({f'{k}-acc_bit_row':acc_bit_row})

    if k == 'w':
        avg_bit_accuracy = correct / cnt
        return avg_bit_accuracy
    else:
        return acc_bit_row_list

def eval_GS(reversed_noise,k=None,device=None):
    total_elements = reversed_noise.shape[0]*reversed_noise.shape[1]
    cnt = 0
    reversed_noise = (reversed_noise - reversed_noise.mean()) / reversed_noise.std()
    torch.manual_seed(217)
    latent_seed = torch.randint(0, 2, (reversed_noise.shape[1], )).to(device)
    acc_bit_row_list = []
    for row in reversed_noise:
        sign_row = (row > 0).int()
        cnt_row = (sign_row == latent_seed).sum().item()
        cnt += cnt_row
        acc_bit_row = cnt_row / reversed_noise.shape[1]
        acc_bit_row_list.append(acc_bit_row)
        
    if k == 'w':
        proportion = cnt / total_elements
        return proportion
    else:
        return acc_bit_row_list


def attack_numpy(attack_type, attack_percentage, X_num, X_cat, X_num_pre, X_cat_pre, args, i):
    mask_col = None
    print(f'attack type: {attack_type}-attack percentage: {attack_percentage}')
    if attack_type == 'rowdeletion':
        num_rows = X_num.shape[0]
        num_rows_delete = int(num_rows * attack_percentage)
        np.random.seed(i+207)
        rows_delete = np.random.choice(num_rows, num_rows_delete, replace=False)
        X_num = np.delete(X_num, rows_delete, axis=0)
        X_cat = np.delete(X_cat, rows_delete, axis=0)
        if args.with_w == 'treering':
            rows_add = np.random.choice(X_num_pre.shape[0], num_rows_delete, replace=False)
            X_num = np.concatenate([X_num, X_num_pre[rows_add]], axis=0)
            X_cat = np.concatenate([X_cat, X_cat_pre[rows_add]], axis=0)
    elif attack_type == 'coldeletion':
        rows = X_num.shape[0]
        cols = X_num.shape[1] + X_cat.shape[1]
        cols_delete = 1 if attack_percentage == 0.05 else 2 if attack_percentage == 0.1 else 3
        if cols_delete > cols:
            raise ValueError('Number of columns to delete is greater than the number of columns')

        np.random.seed(i+207)
        cols_delete = np.random.choice(cols, cols_delete, replace=False)
        mask_col = cols_delete
        for index in cols_delete:
            if index < X_num.shape[1]:
                X_num[:, index] = X_num_pre[:rows, index]
            else:
                X_cat[:, index - X_num.shape[1]] = X_cat_pre[:rows, index - X_num.shape[1]]
                
    elif attack_type == 'celldeletion':
        num_values = X_num.shape[0] * X_num.shape[1] + X_cat.shape[0] * X_cat.shape[1]
        num_values_delete = int(num_values * attack_percentage)
        np.random.seed(i+207)
        values_delete = np.random.choice(num_values, num_values_delete, replace=False)
        rows_delete = values_delete // (X_num.shape[1]+X_cat.shape[1])
        cols_delete = values_delete % (X_num.shape[1]+X_cat.shape[1])
        for i, index in enumerate(cols_delete):
            if index < X_num.shape[1]:
                X_num[rows_delete[i], index] = X_num_pre[rows_delete[i], index]
            else :
                X_cat[rows_delete[i], index - X_num.shape[1]] = X_cat_pre[rows_delete[i], index - X_num.shape[1]]
        
                
    elif attack_type == 'noise':
        np.random.seed(i+207)
        # multiplier = np.random.uniform(1-attack_percentage, 1+attack_percentage, X_num.shape) 

        # X_num = X_num * multiplier
        
        noise = np.random.normal(
            loc=0.0,
            scale=attack_percentage * np.abs(X_num),
            size=X_num.shape
        )
        X_num = X_num + noise
        
    elif attack_type == 'shuffle':
        np.random.seed(i+207)
        rows_shuffle = np.random.permutation(X_num.shape[0])
        X_num = X_num[rows_shuffle]
        X_cat = X_cat[rows_shuffle]
        
    elif attack_type == 'catsubstitution':
        np.random.seed(i+207)
        num_values = X_cat.shape[0] * X_cat.shape[1]
        num_values_delete = int(num_values * attack_percentage)
        values_delete = np.random.choice(num_values, num_values_delete, replace=False)
        rows_delete = values_delete // (X_cat.shape[1])
        cols_delete = values_delete % (X_cat.shape[1])
        X_cat[rows_delete, cols_delete] = X_cat_pre[rows_delete, cols_delete]
        
    elif attack_type == 'precision':
        # limiting numerical precision (rounding) - only applies to X_num
        # attack_percentage: 0.2->round at 1st significant digit, 0.1->round at 2nd, 0.05->round at 3rd
        np.random.seed(i+207)
        
        # Map attack_percentage to significant digit positions to round at
        if attack_percentage == 0.2:
            round_at_digit = 1  # Round at 1st significant digit
        elif attack_percentage == 0.1:
            round_at_digit = 2  # Round at 2nd significant digit
        elif attack_percentage == 0.05:
            round_at_digit = 3  # Round at 3rd significant digit
        else:
            round_at_digit = max(1, int(1.0 / attack_percentage))
        
        # Create masks for efficient processing
        has_decimal = (X_num != 0) & (np.abs(X_num) != np.abs(np.floor(X_num)))
        
        # Process only values with decimal parts
        decimal_values = X_num[has_decimal]
        
        if len(decimal_values) > 0:
            # Convert to string representation for analysis
            abs_values = np.abs(decimal_values)
            
            # Use vectorized operations where possible
            for idx, value in enumerate(decimal_values):
                # Get the actual indices in the original array
                indices = np.where(has_decimal)
                row_idx, col_idx = indices[0][idx], indices[1][idx]
                
                # Analyze decimal part
                value_str = f"{abs(value):.15f}".rstrip('0')
                
                if '.' in value_str:
                    decimal_part = value_str.split('.')[1]
                    
                    # Find first non-zero digit position
                    first_nonzero_pos = next((i for i, digit in enumerate(decimal_part) if digit != '0'), -1)
                    
                    if first_nonzero_pos >= 0:
                        # Count significant digits after first non-zero
                        significant_digits = decimal_part[first_nonzero_pos:]
                        
                        # Only process if there are 3 or more significant digits
                        if len(significant_digits) >= 3:
                            # Calculate decimal places to truncate to
                            truncate_to_decimals = first_nonzero_pos + round_at_digit
                            
                            # Use truncation instead of rounding to preserve exact significant digits
                            multiplier = 10 ** truncate_to_decimals
                            truncated_value = np.trunc(value * multiplier) / multiplier
                            X_num[row_idx, col_idx] = truncated_value
        
    elif attack_type == 'stratified_sampling':
        # stratified sub or super-sampling - applies to both X_num and X_cat
        np.random.seed(i+207)
        n_rows = X_num.shape[0]
        
        # Extract target column (last column of X_cat) and get unique values
        target_values = np.unique(X_cat[:, -1])
        s = len(target_values)  # number of unique classes
        
        # Calculate target size for each group
        base_group_size = n_rows // s
        remainder = n_rows % s
        
        # Store indices for the final stratified sample
        final_indices = []
        
        for i, target_val in enumerate(target_values):
            # Find indices for current class
            class_indices = np.where(X_cat[:, -1] == target_val)[0]
            current_class_size = len(class_indices)
            
            # Determine target size for this group
            if i < remainder:
                target_size = base_group_size + 1  # Add 1 to first 'remainder' groups
            else:
                target_size = base_group_size
            
            if current_class_size < target_size:
                # Oversample: randomly replicate samples
                additional_needed = target_size - current_class_size
                additional_indices = np.random.choice(class_indices, additional_needed, replace=True)
                selected_indices = np.concatenate([class_indices, additional_indices])
            elif current_class_size > target_size:
                # Undersample: randomly select samples
                selected_indices = np.random.choice(class_indices, target_size, replace=False)
            else:
                # Exact match
                selected_indices = class_indices
            
            final_indices.extend(selected_indices)
        
        # Convert to numpy array and shuffle
        final_indices = np.array(final_indices)
        np.random.shuffle(final_indices)
        
        assert len(final_indices) == X_num.shape[0]
        # Apply stratified sampling to both X_num and X_cat
        X_num = X_num[final_indices]
        X_cat = X_cat[final_indices]
                     
    elif attack_type == 'quantization':
        # quantization (e.g. through Quantile Transformation) - applies to X_num
        from sklearn.preprocessing import QuantileTransformer
        np.random.seed(i+207)
        
        # attack_percentage determines number of quantiles
        n_quantiles = max(2, int(1.0 / attack_percentage))
        
        for col_idx in range(X_num.shape[1]):
            col_data = X_num[:, col_idx].reshape(-1, 1)
            qt = QuantileTransformer(n_quantiles=n_quantiles, random_state=i+207)
            # Transform to quantiles and back to discretize the data
            transformed = qt.fit_transform(col_data)
            # Discretize by rounding to create quantile bins
            discretized = np.round(transformed * (n_quantiles - 1)) / (n_quantiles - 1)
            X_num[:, col_idx] = qt.inverse_transform(discretized).flatten()
            
    elif attack_type == 'gaussian_noise_standardized':
        # Gaussian noise injection in standardized space - applies to X_num
        np.random.seed(i+207)
        
        for col_idx in range(X_num.shape[1]):
            col_data = X_num[:, col_idx]
            
            # Standardize: z_ij = (x_ij - μ_j) / σ_j
            mu_j = np.mean(col_data)
            sigma_j = np.std(col_data)
            
            if sigma_j > 0:  # Avoid division by zero
                z_ij = (col_data - mu_j) / sigma_j
                
                # Add Gaussian noise: z'_ij = z_ij + ε * N(0,1)
                epsilon = attack_percentage  # attack_percentage serves as noise intensity
                noise = np.random.normal(0, 1, z_ij.shape)
                z_prime_ij = z_ij + epsilon * noise
                
                # Transform back: x'_ij = z'_ij * σ_j + μ_j
                X_num[:, col_idx] = z_prime_ij * sigma_j + mu_j
        
    else:
        raise ValueError('Attack type not supported')
    return X_num, X_cat, mask_col

def main(args,num_loops):
    X_num_pre = None
    X_cat_pre = None
    syn_data_pre = None
    w_avg_acc_bit = []
    w_avg_acc_bit_num = []
    no_w_acc_bit = []
    no_w_acc_bit_num = []
    for i in range(num_loops):
        if args.with_w in ['TAB-DRW', 'GLW', 'tabmark', 'muse']:
            if i == 0:
                syn_data_pre, avg_bit_accuracy, avg_bit_accuracy_num, no_w_acc_bit_row_list, no_w_acc_bit_row_list_num = loop(args, i, None, None, None)
            else:
                syn_data_pre, avg_bit_accuracy, avg_bit_accuracy_num, no_w_acc_bit_row_list, no_w_acc_bit_row_list_num = loop(args, i, None, None, syn_data_pre)
            no_w_acc_bit_row_list = no_w_acc_bit_row_list.tolist()
            no_w_acc_bit_row_list_num = no_w_acc_bit_row_list_num.tolist()
            w_avg_acc_bit_num.append(avg_bit_accuracy_num)
            no_w_acc_bit_num.extend(no_w_acc_bit_row_list_num)
        else:
            if i == 0:
                X_num_pre, X_cat_pre, avg_bit_accuracy, no_w_acc_bit_row_list = loop(args, i, None, None, None)
            else:
                X_num_pre, X_cat_pre, avg_bit_accuracy, no_w_acc_bit_row_list = loop(args, i, X_num_pre, X_cat_pre, None)
        w_avg_acc_bit.append(avg_bit_accuracy)
        no_w_acc_bit.extend(no_w_acc_bit_row_list)
        
    no_w_acc_bit, no_w_acc_bit_num = np.array(no_w_acc_bit), np.array(no_w_acc_bit_num)
    avg_no_w_acc_bit = np.mean(no_w_acc_bit)
    std_no_w_acc_bit = np.std(no_w_acc_bit, ddof=1)
    
    if args.with_w in ['TAB-DRW', 'GLW', 'tabmark', 'muse']:
        avg_no_w_acc_bit_num = np.mean(no_w_acc_bit_num)
        std_no_w_acc_bit_num = np.std(no_w_acc_bit_num, ddof=1)
        
        if args.with_w == 'tabmark':
            if args.attack == 'rowdeletion':
                Z_score_list = [(10 ** 0.5) * (x - avg_no_w_acc_bit) / (std_no_w_acc_bit / np.sqrt(args.num_samples * (1-args.attack_percentage))) for x in w_avg_acc_bit]
                Z_score_list_num = [(10 ** 0.5) * (x - avg_no_w_acc_bit_num) / (std_no_w_acc_bit_num / np.sqrt(args.num_samples * (1-args.attack_percentage))) for x in w_avg_acc_bit_num]
            else:
                Z_score_list = [(10 ** 0.5) * (x - avg_no_w_acc_bit) / (std_no_w_acc_bit / np.sqrt(args.num_samples)) for x in w_avg_acc_bit]
                Z_score_list_num = [(10 ** 0.5) * (x - avg_no_w_acc_bit_num) / (std_no_w_acc_bit_num / np.sqrt(args.num_samples)) for x in w_avg_acc_bit_num]
        else:   
            if args.attack == 'rowdeletion':
                Z_score_list = [(x - avg_no_w_acc_bit) / (std_no_w_acc_bit / np.sqrt(args.num_samples * (1-args.attack_percentage))) for x in w_avg_acc_bit]
                Z_score_list_num = [(x - avg_no_w_acc_bit_num) / (std_no_w_acc_bit_num / np.sqrt(args.num_samples * (1-args.attack_percentage))) for x in w_avg_acc_bit_num]
            else:
                Z_score_list = [(x - avg_no_w_acc_bit) / (std_no_w_acc_bit / np.sqrt(args.num_samples)) for x in w_avg_acc_bit]
                Z_score_list_num = [(x - avg_no_w_acc_bit_num) / (std_no_w_acc_bit_num / np.sqrt(args.num_samples)) for x in w_avg_acc_bit_num]
            
        print(f'{args.dataname}_{args.with_w}_{args.num_samples} Z-score mean/std: {np.mean(Z_score_list)}/{np.std(Z_score_list)}')
        print(f'{args.dataname}_{args.with_w}_{args.num_samples} Z-score (wm numerical col only) mean/std: {np.mean(Z_score_list_num)}/{np.std(Z_score_list_num)}')
        
        # approximately compute Type I error
        H_0_Z_score_list = [(np.mean(no_w_acc_bit[i*args.num_samples:(i+1)*args.num_samples]) - avg_no_w_acc_bit) / (std_no_w_acc_bit / np.sqrt(args.num_samples)) for i in range(num_loops)]
        H_0_Z_score_list_num = [(np.mean(no_w_acc_bit_num[i*args.num_samples:(i+1)*args.num_samples]) - avg_no_w_acc_bit_num) / (std_no_w_acc_bit_num / np.sqrt(args.num_samples)) for i in range(num_loops)]
        print(f'{args.dataname}_no-w_{args.num_samples} Z-score mean/std: {np.mean(H_0_Z_score_list)}/{np.std(H_0_Z_score_list)}, \
              type I error with significance level {args.alpha}: {compute_detection_rate(H_0_Z_score_list, args.alpha)}')
        print(f'{args.dataname}_no-w_{args.num_samples} Z-score (wm numerical col only) mean/std: {np.mean(H_0_Z_score_list_num)}/{np.std(H_0_Z_score_list_num)}, \
              type I error with significance level {args.alpha}: {compute_detection_rate(H_0_Z_score_list_num, args.alpha)}')
        
    else:
        if args.attack == 'rowdeletion':
            Z_score_list = [(x - avg_no_w_acc_bit) / (std_no_w_acc_bit / np.sqrt(args.num_samples * (1-args.attack_percentage))) for x in w_avg_acc_bit]
        else:
            Z_score_list = [(x - avg_no_w_acc_bit) / (std_no_w_acc_bit / np.sqrt(args.num_samples)) for x in w_avg_acc_bit]
        print(f'{args.dataname}_{args.with_w}_{args.num_samples} Z-score mean/std: {np.mean(Z_score_list)}/{np.std(Z_score_list)}')
        
        H_0_Z_score_list = [(np.mean(no_w_acc_bit[i*args.num_samples:(i+1)*args.num_samples]) - avg_no_w_acc_bit) / (std_no_w_acc_bit / np.sqrt(args.num_samples)) for i in range(num_loops)]
        print(f'{args.dataname}_no-w_{args.num_samples} Z-score mean/std: {np.mean(H_0_Z_score_list)}/{np.std(H_0_Z_score_list)}, \
              type I error with significance level {args.alpha}: {compute_detection_rate(H_0_Z_score_list, args.alpha)}')
        
def loop(args, i, X_num_pre=None, X_cat_pre=None, syn_data_pre=None):
    dataname = args.dataname
    device = args.device
    save_path_arg = args.save_path
    w_radius = args.w_radius
    with_w = args.with_w

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    num_samples = args.num_samples
    
    save_dir = f'{curr_dir}/../{save_path_arg}/{with_w}/{num_samples}/{i}'
    save_dir_real_data = f'{curr_dir}/../{save_path_arg}'

    if not os.path.exists(save_dir):
        # If it doesn't exist, create it
        os.mkdir(save_dir)

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse, _ = get_input_generate(args)
    in_dim = train_z.shape[1]

    try:
        watermarking_mask = torch.tensor(np.load(f'{save_dir}/watermarking_mask.npy')).to(device)
    except:
        watermarking_mask = None

    try:
        gt_patch = torch.tensor(np.load(f'{save_dir}/gt_patch.npy')).to(device)
    except:
        gt_patch = None
    mean = train_z.mean(0)
    # Loading diffusion model for inverse process
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    model = DDIMModel(denoise_fn).to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))
    else:
        model.load_state_dict(torch.load(f'{ckpt_path}/model.pt', map_location=torch.device('cpu')))
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    X_num = None
    X_cat = None
    mask_col = None
    avg_bit_accuracy = None
    avg_bit_accuracy_num = None
    no_w_acc_bit_row_list = None
    no_w_acc_bit_row_list_num = None
    # get the latent of the synthetic tabular from the vae encoder
    if args.mode == 'detect':
        pre_keys = ['no-w', 'w', 'w-num']
        if i in range(100):
            if with_w in ['TAB-DRW', 'GLW', 'tabmark', 'muse']:
                for k in pre_keys:
                    if k == 'w-num' and with_w == 'muse':
                        save_path = f'{save_dir}/w-{args.method}-raw.csv'
                    else:
                        save_path = f'{save_dir}/{k}-{args.method}-raw.csv'
                        
                    save_path = f'{save_dir}/{k}-{args.method}-raw.csv'
                    save_path_ori = f'{save_dir}/no-w-tabsyn-raw.csv'
                    save_path_real_data = f'{save_dir_real_data}/real_raw.csv'
                    # df = pd.read_csv(save_path)
                    
                    try:
                        syn_data = np.loadtxt(save_path, delimiter=',', skiprows=1)
                        syn_data_ori = np.loadtxt(save_path_ori, delimiter=',', skiprows=1)
                        real_data = np.loadtxt(save_path_real_data, delimiter=',', skiprows=1)
                    except ValueError:
                        data = np.genfromtxt(
                            save_path,
                            delimiter=',',
                            skip_header=1,
                            dtype=float,
                            missing_values='',
                            filling_values=np.nan
                        )
                        for col in range(data.shape[1]):
                            col_vals = data[:, col]
                            valid = col_vals[~np.isnan(col_vals)]
                            if valid.size == 0:
                                continue
                            uniq, counts = np.unique(valid, return_counts=True)
                            mode_val = uniq[np.argmax(counts)]
                            data[np.isnan(col_vals), col] = mode_val
                        syn_data = data
                    
                    syn_data_pre = syn_data.copy() if syn_data_pre is None else syn_data_pre
                    
                    task_type = info['task_type']
                    num_col_idx = info['num_col_idx'].copy()
                    cat_col_idx = info['cat_col_idx'].copy()
                    target_col_idx = info['target_col_idx']
                    key_col_idx = info['key_col_idx'].copy()
                    value_col_idx = info['value_col_idx'].copy()
                    dis_col_idx = info['dis_col_idx'].copy()
                    continuous_col_idx = info['continuous_col_idx'].copy()
                    
                    if task_type == 'regression':
                        num_col_idx.extend(target_col_idx)
                    else:
                        cat_col_idx.extend(target_col_idx)
                        
                    X_num = syn_data[:, num_col_idx]
                    X_cat = syn_data[:, cat_col_idx]
                    X_num_pre = syn_data_pre[:, num_col_idx]
                    X_cat_pre = syn_data_pre[:, cat_col_idx]
                    
                    # attack
                    if args.attack != 'none' and (k == 'w' or k == 'w-num'):
                        X_num, X_cat, mask_col = attack_numpy(args.attack, args.attack_percentage, X_num, X_cat, X_num_pre, X_cat_pre, args, i)
                        if args.attack == 'rowdeletion':
                            syn_data[:X_num.shape[0], num_col_idx] = X_num
                            syn_data[:X_cat.shape[0], cat_col_idx] = X_cat
                            syn_data = syn_data[:X_num.shape[0]]
                        else:
                            syn_data[:, num_col_idx] = X_num
                            syn_data[:, cat_col_idx] = X_cat
                    
    
                    if k == 'w':
                        start_time = time.time()
                        if with_w == 'GLW':
                            avg_bit_accuracy = eval_GLW(syn_data, k, i)
                        elif with_w == 'tabmark':
                            avg_bit_accuracy = eval_tabmark(syn_data, k, syn_data_ori, continuous_col_idx[0], out_seed=i)
                        elif with_w == 'muse':
                            avg_bit_accuracy = eval_muse(syn_data, k, real_data, out_seed=i)
                        else:
                            avg_bit_accuracy = eval_TAB_DRW(syn_data, with_w, k, i, num_col_idx, cat_col_idx)
                        end_time = time.time()
                        print(f'running time: {end_time - start_time}')
                        print(f'{k}:', avg_bit_accuracy)
                    elif k == 'w-num':
                        if with_w == 'GLW':
                            avg_bit_accuracy_num = eval_GLW(syn_data[:, value_col_idx], k, i)
                        elif with_w == 'tabmark':
                            avg_bit_accuracy_num = eval_tabmark(syn_data, k, syn_data_ori, continuous_col_idx[0], out_seed=i)
                        elif with_w == 'muse':
                            avg_bit_accuracy_num = eval_muse(syn_data, k, real_data, out_seed=i)
                        else:
                            avg_bit_accuracy_num = eval_TAB_DRW(syn_data, with_w, k, i, num_col_idx, cat_col_idx, value_col_idx)
                        print(f'{k}:', avg_bit_accuracy_num)
                    else:
                        discrete = np.array([True if i in cat_col_idx or i in dis_col_idx else False for i in range(syn_data.shape[1])])
                        discrete_num = np.array([True if num_col_idx[i] in dis_col_idx else False for i in range(syn_data[:, value_col_idx].shape[1])])
                        if with_w == 'GLW':
                            no_w_acc_bit_row_list = eval_GLW(syn_data, k, i, discrete)
                            no_w_acc_bit_row_list_num = eval_GLW(syn_data[:, value_col_idx], k, i, discrete_num)
                        elif with_w == 'tabmark':
                            no_w_acc_bit_row_list = eval_tabmark(syn_data, k, syn_data_ori, continuous_col_idx[0], out_seed=i)
                            no_w_acc_bit_row_list_num = eval_tabmark(syn_data, k, syn_data_ori, continuous_col_idx[0], out_seed=i)
                        elif with_w == 'muse':
                            no_w_acc_bit_row_list = eval_muse(syn_data, k, real_data, out_seed=i)
                            no_w_acc_bit_row_list_num = eval_muse(syn_data, k, real_data, out_seed=i)
                        else:
                            no_w_acc_bit_row_list = eval_TAB_DRW(syn_data, with_w, k, i, num_col_idx, cat_col_idx, np.arange(syn_data.shape[1]))
                            no_w_acc_bit_row_list_num = eval_TAB_DRW(syn_data, with_w, k, i, num_col_idx, cat_col_idx, value_col_idx)
                        # no_w_acc_bit_row_list_num = eval_TAB_DRW_pair(syn_data, with_w, k, i, value_col_idx, key_col_idx) if with_w == 'TAB-DRW_pair' and args.w_value_percent != 1.0 else eval_TAB_DRW(syn_data, with_w, k, num_col_idx)
                        print(f'{k}:', np.mean(no_w_acc_bit_row_list), np.std(no_w_acc_bit_row_list, ddof=1))
                        print(f'{k}-num:', np.mean(no_w_acc_bit_row_list_num), np.std(no_w_acc_bit_row_list_num, ddof=1))
                    
                    # wandb.log({f'{k}':float(metric)})
            else:
                for k in pre_keys[:-1]:
                    save_path = f'{save_dir}/{k}-{args.method}.csv'
                    latents = torch.tensor(np.load(f'{save_dir}/{k}-{args.method}.npy')).to(device)
                    X_num = torch.tensor(np.load(f'{save_dir}/{k}-{args.method}-X_num.npy')).to(device)
                    X_cat = torch.tensor(np.load(f'{save_dir}/{k}-{args.method}-X_cat.npy')).to(device)
                    # load X_num, X_cat from the disk
                    X_num_pre = X_num if X_num_pre is None else X_num_pre
                    X_cat_pre = X_cat if X_cat_pre is None else X_cat_pre
                    # attack
                    if args.attack != 'none' and k == 'w':
                        X_num, X_cat, mask_col = attack_numpy(args.attack, args.attack_percentage, X_num.cpu().numpy(), X_cat.cpu().numpy(), X_num_pre.cpu().numpy(), X_cat_pre.cpu().numpy(), args, i)
                        X_num = torch.tensor(X_num).to(device)
                        X_cat = torch.tensor(X_cat).to(device)
                    
                    if k == 'w':
                        avg_bit_accuracy = get_watermark_metric(args, dataname, save_path, save_dir, k, info, model, noise_scheduler, watermarking_mask, gt_patch, mean, latents, X_num, X_cat,k, mask_col)
                        print(f'{k}:', avg_bit_accuracy)
                    else:
                        no_w_acc_bit_row_list = get_watermark_metric(args, dataname, save_path, save_dir, k, info, model, noise_scheduler, watermarking_mask, gt_patch, mean, latents, X_num, X_cat,k, mask_col)
                        print(f'{k}:', np.mean(no_w_acc_bit_row_list), np.std(no_w_acc_bit_row_list, ddof=1))
                    # wandb.log({f'{k}':float(metric)})
                    
        if with_w in ['TAB-DRW', 'GLW', 'tabmark', 'muse']:
            return syn_data, avg_bit_accuracy, avg_bit_accuracy_num, no_w_acc_bit_row_list, no_w_acc_bit_row_list_num
        else:
            return X_num, X_cat, avg_bit_accuracy, no_w_acc_bit_row_list


if __name__ == '__main__':
    pass
