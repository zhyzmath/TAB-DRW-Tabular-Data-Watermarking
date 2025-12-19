import os
import sys
import json
import argparse
from typing import List

import numpy as np
import pandas as pd


def load_info(info_path: str) -> dict:
    with open(info_path, 'r') as f:
        return json.load(f)


def ensure_repo_root_in_path(repo_root: str) -> None:
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def encode_real_to_raw_for_dataset(repo_root: str, dataname: str) -> bool:
    """
    Encode categorical columns (and target for classification) in synthetic/{dataname}/real.csv
    into raw indices and save to real_raw.csv.
    Reuses utils_train.preprocess cat_forward to match the training-time mapping.
    Returns whether processing succeeded.
    """
    dataset_dir = os.path.join(repo_root, 'data', dataname)
    info_path = os.path.join(dataset_dir, 'info.json')
    real_csv_path = os.path.join(repo_root, 'synthetic', dataname, 'real.csv')
    out_csv_path = os.path.join(repo_root, 'synthetic', dataname, 'real_raw.csv')

    if not os.path.exists(real_csv_path):
        print(f'[skip] {dataname}: {real_csv_path} not found')
        return False
    if not os.path.exists(info_path):
        print(f'[skip] {dataname}: {info_path} not found')
        return False

    info = load_info(info_path)
    task_type = info['task_type']

    # Delay-import project modules to keep the CLI lightweight.
    from utils_train import preprocess

    # Get forward encoder (training-time category mapping).
    _, _, _, _, _, cat_inverse, cat_forward = preprocess(dataset_dir, task_type=task_type, inverse=True)

    df = pd.read_csv(real_csv_path)

    # Drop rows that contain any NaN.
    orig_len = len(df)
    df = df.dropna()
    dropped = orig_len - len(df)
    if dropped > 0:
        print(f'[info] {dataname}: dropped {dropped} rows containing NaN (from {orig_len})')
    if len(df) == 0:
        # All rows dropped; write an empty file.
        df.to_csv(out_csv_path, index=False)
        print(f'[ok] {dataname}: all rows contained NaN, wrote empty file {out_csv_path}')
        return True

    cat_col_idx: List[int] = info.get('cat_col_idx', [])
    target_col_idx: List[int] = info['target_col_idx']

    # Only classification tasks need target encoding.
    encode_target = task_type in ['binclass', 'multiclass'] and len(target_col_idx) == 1

    # Build input for cat_forward: [y | X_cat]
    if encode_target:
        y_arr = df.iloc[:, target_col_idx[0]].to_numpy().reshape(-1, 1)
    else:
        # For regression, y is not encoded; pass an empty array for shape compatibility.
        y_arr = np.empty((df.shape[0], 0), dtype=object)

    x_cat_arr = df.iloc[:, cat_col_idx].to_numpy() if len(cat_col_idx) else np.empty((df.shape[0], 0), dtype=object)
    x_cat_concat = np.concatenate([y_arr, x_cat_arr], axis=1).astype(object)

    if x_cat_concat.shape[1] == 0:
        # No columns to encode (e.g., all numeric regression); copy as-is.
        df.to_csv(out_csv_path, index=False)
        print(f'[ok] {dataname}: no encoding needed, wrote {out_csv_path}')
        return True

    x_cat_encoded = cat_forward(x_cat_concat)

    # Split back into y and X_cat.
    if encode_target:
        y_encoded = x_cat_encoded[:, 0].astype(np.int64)
        x_cat_encoded_only = x_cat_encoded[:, 1:].astype(np.int64)
        df.iloc[:, target_col_idx[0]] = y_encoded
        if len(cat_col_idx):
            df.iloc[:, cat_col_idx] = x_cat_encoded_only
    else:
        # No target encoding; only overwrite categorical features.
        if len(cat_col_idx):
            df.iloc[:, cat_col_idx] = x_cat_encoded.astype(np.int64)

    df.to_csv(out_csv_path, index=False)
    print(f'[ok] {dataname}: wrote {out_csv_path}')
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert real.csv categorical columns to raw indices for multiple datasets.')
    parser.add_argument('datasets', nargs='*', default=['adult', 'magic', 'shoppers', 'default', 'drybean'], help='Dataset names under synthetic/. Default: %(default)s')
    args = parser.parse_args()

    # Infer repo root from script location (parent of synthetic).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    ensure_repo_root_in_path(repo_root)

    any_ok = False
    for dataname in args.datasets:
        try:
            ok = encode_real_to_raw_for_dataset(repo_root, dataname)
            any_ok = any_ok or ok
        except Exception as e:
            print(f'[error] {dataname}: {e}')

    if not any_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()

