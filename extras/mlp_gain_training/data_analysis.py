#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def constant_mask(arr, eps_var=1e-8, abs_ratio=1e-3, cv_ratio=5e-4):
    std_all  = arr.std(axis=0)
    mean_all = arr.mean(axis=0)
    hard_zero = std_all < eps_var
    med_std   = np.median(std_all)
    near_zero = std_all < abs_ratio * med_std
    cv        = std_all / np.maximum(np.abs(mean_all), 1e-9)
    low_cv    = cv < cv_ratio
    drop_mask = hard_zero | (near_zero & low_cv) | low_cv
    keep_mask = ~drop_mask
    return keep_mask

def compute_stats(df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    means = df.mean()
    stds  = df.std(ddof=1)
    stats = pd.DataFrame({'mean': means, 'std': stds})
    if output_path:
        stats.to_csv(output_path, index_label='column')
    return stats

def main():
    input_csv  = 'gru_training_data.csv'
    output_csv = 'gru_training_data_stats.csv'

    # 1. 讀入原始資料
    df = pd.read_csv(input_csv)

    # 2. 計算 mean & std
    stats_df = compute_stats(df, output_csv)

    # 3. 計算 keep/drop mask
    mask = constant_mask(
        arr=df.values,
        eps_var=1e-8,
        abs_ratio=1e-3,
        cv_ratio=5e-4
    )
    stats_df['keep'] = mask

    # 4. 依照原始欄位順序，重設 index (0,1,2…)，並把原欄名搬進 column 這欄
    result = (
        stats_df
        .reset_index()                # 把原本的 index (欄位名) 變成一個欄位
        .rename(columns={'index':'column'})
        .reset_index(drop=True)       # 重新從 0 開始編號
    )

    # 5. pandas 顯示設定
    pd.set_option('display.max_rows',    None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width',       None)

    # 6. 列印最終結果
    print("各欄位統計結果（依原 CSV 順序，keep=True/False 混排）：")
    print(result)

if __name__ == '__main__':
    main()
