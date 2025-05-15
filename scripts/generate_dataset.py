#!/usr/bin/env python
"""使用原 Kalman 解析解 (K = P Hᵀ (H P Hᵀ + V)⁻¹) 生成 (x → K) 樣本"""
import argparse, numpy as np

def kalman_gain(P, H, V):
    M = H @ P @ H.T + V
    return P @ H.T @ np.linalg.inv(M)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100000)
    ap.add_argument("--out",     type=str, required=True)
    args = ap.parse_args()

    H = np.eye(2, 6, dtype=np.float32)      # 觀測僅量測 x,y
    V = np.eye(2, dtype=np.float32) * 0.01  # 假設量測誤差
    P = np.eye(6, dtype=np.float32) * 0.1   # 先驗共變異

    x = np.random.uniform(-1, 1, (args.samples, 6)).astype(np.float32)
    y = np.empty((args.samples, 12), np.float32)  # 6×2 展平

    for i in range(args.samples):
        K = kalman_gain(P, H, V).astype(np.float32)
        y[i] = K.flatten()

    np.savez_compressed(args.out, x=x, y=y)

if __name__ == "__main__":
    main()
