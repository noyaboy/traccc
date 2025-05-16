#!/usr/bin/env python
"""使用 Kalman 解析解 (K = P Hᵀ (H P Hᵀ + V)⁻¹) 生成 (x → K) 樣本

改進 v2
-------
整合先前「潛在不足」的完整修正：

1. **狀態範圍拓寬＋Long-tail 外點**  
   - 80 %：x ∼ U(−1, 1)  
   - 15 %：x ∼ U(−2, 2)（|x| > 1 區段）  
   - 5 % ：x ∼ 𝓝(0, σ = 3)（截斷於 ±10）  

2. **先驗協方差 P 含弱相關項**  
   - 對角元素仍依 |x| 調整  
   - 10 % 機率於隨機項對加入 ρ ∈ [−0.3, 0.3]  
   - 以相關矩陣法組裝，並以最小特徵值修正確保正定  

3. **量測雜訊 V 加入 x–y 相關**  
   - Varₓ, Varᵧ ∼ U(0.005, 0.05)  
   - 50 % 機率加入 ρ ∈ [−0.5, 0.5] 的相關  

4. **介面向下相容**  
   - 仍固定僅觀測位置 (H ∈ ℝ^{2×6})；`y.shape==(N,12)` 不變  
   - 下游訓練程式無須修改  

（動態 Q、可變 H 之進階場景留待後續版本）"""

import argparse
import numpy as np

def kalman_gain(P: np.ndarray, H: np.ndarray, V: np.ndarray) -> np.ndarray:
    """K = P Hᵀ (H P Hᵀ + V)⁻¹"""
    M = H @ P @ H.T + V
    return P @ H.T @ np.linalg.inv(M)


def random_state_vector(rng: np.random.Generator) -> np.ndarray:
    """6-D mixture distribution：Uniform + wider range + truncated Gaussian."""
    r = rng.random()
    if r < 0.80:            # 80 %
        return rng.uniform(-1, 1, 6)
    elif r < 0.95:          # 15 %
        return rng.uniform(-2, 2, 6)
    else:                   # 5 % long-tail Gaussian
        x = rng.normal(0.0, 3.0, 6)
        return np.clip(x, -10, 10)


def random_prior_cov(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """SPD prior P，約 10 % off-diag 相關。"""
    diag_P = 0.05 + 0.15 * np.abs(x) + rng.uniform(0.0, 0.02, 6)

    C = np.eye(6, dtype=np.float32)             # correlation matrix
    for i in range(6):
        for j in range(i):
            if rng.random() < 0.10:             # 10 % 機率加入相關
                rho = rng.uniform(-0.30, 0.30)
                C[i, j] = C[j, i] = rho

    # 以 λ_min 修正，確保正定
    lam_min = np.linalg.eigvalsh(C).min()
    if lam_min < 0.05:
        C += (0.05 - lam_min + 1e-6) * np.eye(6, dtype=np.float32)

    sqrt_d = np.sqrt(diag_P).astype(np.float32)
    return (sqrt_d[:, None] * C * sqrt_d[None, :]).astype(np.float32)


def random_measurement_cov(rng: np.random.Generator) -> np.ndarray:
    """2×2 SPD V，含可選相關。"""
    var = rng.uniform(0.005, 0.05, 2).astype(np.float32)
    rho = rng.uniform(-0.5, 0.5) if rng.random() < 0.5 else 0.0
    off = rho * np.sqrt(var[0] * var[1])
    return np.array([[var[0], off],
                     [off,    var[1]]], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100_000)
    ap.add_argument("--out",     type=str, required=True)
    args = ap.parse_args()

    H = np.eye(2, 6, dtype=np.float32)  # observe x, y only

    rng = np.random.default_rng()
    x = np.empty((args.samples, 6), dtype=np.float32)
    y = np.empty((args.samples, 12), dtype=np.float32)  # K flattened (6×2)

    for i in range(args.samples):
        xi = random_state_vector(rng).astype(np.float32)
        P  = random_prior_cov(xi, rng)
        V  = random_measurement_cov(rng)

        K = kalman_gain(P, H, V).astype(np.float32)

        x[i] = xi
        y[i] = K.flatten()

    np.savez_compressed(args.out, x=x, y=y)
    print(f"Saved dataset with {args.samples} samples to {args.out}")

if __name__ == "__main__":
    main()
