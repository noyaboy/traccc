#!/usr/bin/env python
"""使用原 Kalman 解析解 (K = P Hᵀ (H P Hᵀ + V)⁻¹) 生成 (x → K) 樣本"""
#!/usr/bin/env python
"""使用 Kalman 解析解 (K = P Hᵀ (H P Hᵀ + V)⁻¹) 生成 (x → K) 樣本

改進重點
---------
1. **每筆樣本皆有不同的先驗協方差 P 與量測雜訊 V**  
   - P 依據輸入狀態向量 x 的幅度動態調整。  
   - 另加入隨機抖動，讓 P 與 V 保持正定且多樣。
2. **保持量測矩陣 H 不變**（僅量測 x、y），避免模型需要額外輸入。  
3. 仍維持 x.shape==(N,6) 與 y.shape==(N,12) 的介面，與既有訓練程式相容。"""

import argparse, numpy as np

def kalman_gain(P, H, V):
    M = H @ P @ H.T + V
    return P @ H.T @ np.linalg.inv(M)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100000)
    ap.add_argument("--out",     type=str, required=True)
    args = ap.parse_args()

    # 固定量測矩陣：僅觀測位置 (state 0:x, 1:y)
    H = np.eye(2, 6, dtype=np.float32)

    rng = np.random.default_rng()

    x = rng.uniform(-1, 1, (args.samples, 6)).astype(np.float32)
    y = np.empty((args.samples, 12), dtype=np.float32)  # 每筆 K 展平成 12 個值

    for i in range(args.samples):
        xi = x[i]

        # --- 動態產生 P ---
        # 先驗不確定度與 |x| 成正比，再加上小幅隨機雜訊 (確保正定)
        diag_P = 0.05 + 0.15 * np.abs(xi) + rng.uniform(0.0, 0.02, 6)
        P = np.diag(diag_P.astype(np.float32))

        # --- 動態產生 V ---
        # 量測雜訊在 [0.005, 0.05] 之間隨機
        diag_V = rng.uniform(0.005, 0.05, 2).astype(np.float32)
        V = np.diag(diag_V)

        K = kalman_gain(P, H, V).astype(np.float32)
        y[i] = K.flatten()

    np.savez_compressed(args.out, x=x, y=y)

if __name__ == "__main__":
    main()