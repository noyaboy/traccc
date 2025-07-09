#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_gru_training_data.py

步驟：
1. 讀取 gru_training_data.csv
2. 只保留 70 欄的列，且所有欄位皆可轉成 float
3. 以 Boxplot (IQR) 檢查離群值，違規整列剔除
4. 以 3 σ 檢查離群值，違規整列剔除
5. 存回 ../../gru_training_data.csv
"""

from pathlib import Path
import csv
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def load_70col_numeric_rows(path: Path) -> pd.DataFrame:
    """只讀取 70 欄且全為數字的列"""
    good_rows = []
    with path.open(newline="") as fh:
        rdr = csv.reader(fh)
        for row in rdr:
            if len(row) != 70:
                continue
            try:
                good_rows.append([float(x) for x in row])
            except ValueError:
                continue  # 有非數字欄則捨棄整列
    return pd.DataFrame(good_rows)


def drop_by_kde_filter(df: pd.DataFrame,
                       bandwidth: float = None,
                       density_thresh: float = 0.05) -> pd.DataFrame:
    """
    一級 KDE 過濾：
      1. 對每一欄做高斯 KDE
      2. 計算每筆值的密度 estimate
      3. 將密度低於 density_thresh 分位之列剔除
    """
    keep = np.ones(len(df), dtype=bool)
    for col in df.columns:
        data = df[col].values
        kde = gaussian_kde(data, bw_method=bandwidth)
        densities = kde.evaluate(data)
        thresh = np.percentile(densities, density_thresh * 100)
        keep &= (densities >= thresh)
        if not keep.any():
            break
    return df.loc[keep]


def drop_by_iqr_filter(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """
    Boxplot (IQR) 過濾：
      1. 計算每欄的 Q1, Q3 與 IQR = Q3-Q1
      2. 剔除超出 [Q1 - k*IQR, Q3 + k*IQR] 的列
    k: IQR 倍數, 預設 1.5
    """
    keep = np.ones(len(df), dtype=bool)
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        keep &= df[col].between(lower, upper)
        if not keep.any():
            break
    return df.loc[keep]


def drop_by_modified_z_score_filter(df: pd.DataFrame, threshold: float = 11.31) -> pd.DataFrame:
    """
    改良 Z–score 過濾 (基於 MAD)：
      1. 對每欄計算 median 與 MAD = median(|x - median|)
      2. 計算改良 Z–score = 0.6745*(x - median)/MAD
      3. 若 |改良 Z–score| > threshold，剔除該列
    threshold: 建議預設 3.5
    """
    keep = np.ones(len(df), dtype=bool)
    for col in df.columns:
        data = df[col].values
        med = np.median(data)
        mad = np.median(np.abs(data - med))
        if mad == 0:
            # MAD 為 0 時，此欄不做過濾
            continue
        mod_z = 0.6745 * (data - med) / mad
        keep &= np.abs(mod_z) <= threshold
        if not keep.any():
            break
    return df.loc[keep]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    in_path = script_dir / "gru_training_data.csv"
    out_path = (script_dir / ".." / ".." / "gru_training_data.csv").resolve()

    # 1-2) 初步過濾
    df = load_70col_numeric_rows(in_path)
    if df.empty:
        print("沒有符合 70 欄且全數字的列。")
        return

    # # 3) Boxplot (IQR) 過濾
    # df = drop_by_iqr_filter(df)
    # if df.empty:
    #     print("所有列在 Boxplot 過濾後被剔除。")
    #     return

    # 4) 改良 Z–score (基於 MAD) 過濾
    df = drop_by_modified_z_score_filter(df)
    if df.empty:
        print("所有列在改良 Z–score 過濾後被剔除。")
        return

    # 5) 輸出
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, header=False)
    print(f"已輸出 {len(df)} 列至 {out_path}")


if __name__ == "__main__":
    main()
