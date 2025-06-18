#!/usr/bin/env python3

"""Train an MLP to approximate the Kalman gain.

The script expects a CSV file `gru_training_data.csv containing rows of
58 input features followed by 12 target values. The first 58 columns encode
`predicted_vec (6), predicted_cov (36), H (12) and V (4).
The remaining 12 columns correspond to the flattened 6x2 Kalman gain matrix.

The training procedure consists of two stages:

1. FP32 pre-training to obtain good initial weights.
2. Quantisation aware training (QAT) to fine tune an INT8 friendly model.

The script saves `model_fp32.pth and model_int8.pt in the output
folder.  Normalisation statistics for both the inputs and targets are stored
alongside the FP32 model and are applied during inference. Targets are
standardised during training and predictions are unnormalised back to the
original scale when reported.
"""
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path
import math
from typing import Tuple
import random
import copy
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.ao.quantization as quant    # ← 官方 QAT 入口
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset

# Data Layout
# std::vector<traccc::scalar> row;
# row.reserve(6 + 36 + 6 * D + D * D + 6 * D);
# for (size_type i = 0; i < 6; ++i) {
#     row.push_back(getter::element(predicted_vec, i, 0));
# }
# for (size_type r = 0; r < 6; ++r) {
#     for (size_type c = 0; c < 6; ++c) {
#         row.push_back(getter::element(predicted_cov, r, c));
#     }
# }
# for (size_type r = 0; r < D; ++r) {
#     for (size_type c = 0; c < 6; ++c) {
#         row.push_back(getter::element(H, r, c));
#     }
# }
# for (size_type r = 0; r < D; ++r) {
#     for (size_type c = 0; c < D; ++c) {
#         row.push_back(getter::element(V, r, c));
#     }
# }
# for (size_type r = 0; r < 6; ++r) {
#     for (size_type c = 0; c < D; ++c) {
#         row.push_back(getter::element(K, r, c));
#     }
# }
# gru_training_logger::write_row(row);


# ───────────── 常數 feature 索引 ─────────────
# ※ 索引為 **0-based**，與 Pandas / NumPy 欄序相同
#   - 前 58 個輸入特徵中共有 35 個常數欄  
#   - 後 12 個目標（Kalman gain）特徵中共有 2 個常數欄
CONST_INPUT_IDXS  = [
    # 原有常數欄位 + 新增 12,18,19,24,25,26,30,31,32,33
    5, 11, 12, 17, 18, 19, 23, 24, 25, 26, 29, 30, 31, 32, 33,
    *range(35, 41),                   # AJ ~ AO
    *range(42, 54),                   # AQ ~ BB
    55, 56                            # BD, BE
]
CONST_OUTPUT_IDXS = [10, 11]          # BQ, BR  → 在 y 張量中的索引

def init_weights(m):
    if isinstance(m, nn.Linear):
        # He 初始化，适用于 ReLU 激活
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        # 如果有偏置，则初始化为 0
        if m.bias is not None:
            init.zeros_(m.bias)

def load_dataset(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load CSV dataset and split inputs/targets."""
    # 讀取 CSV 並移除行數不是 70 的 row
    df = pd.read_csv(path, header=None, usecols=range(70))
    orig_rows = len(df)
    # drop rows with missing values → 非 70 欄的 row
    df = df.dropna(how='any')
    arr = df.values.astype("float32")
    final_rows = arr.shape[0]
    print(f"原始 row 數: {orig_rows}, 篩選後 row 數: {final_rows}")
    # 不再做 ±3*標準差過濾
    data = arr
    # ────── 移除常數輸入欄位 ──────
    kept_input_cols = [i for i in range(58) if i not in CONST_INPUT_IDXS]
    x = torch.tensor(data[:, kept_input_cols])

    # ────── 移除常數輸出欄位 ──────
    y_full = torch.tensor(data[:, 58:])          # shape: (N, 12)
    kept_output_cols = [i for i in range(12) if i not in CONST_OUTPUT_IDXS]
    y = y_full[:, kept_output_cols]

    print(f"移除常數特徵後，x_dim={x.shape[1]}, y_dim={y.shape[1]}")
    return x, y
    
def split_data(x: torch.Tensor, y: torch.Tensor, train_ratio=0.8, val_ratio=0.1):
    """Shuffle and split the data set."""
    idx = torch.randperm(x.shape[0])
    x = x[idx]
    y = y[idx]

    n_train = int(train_ratio * x.shape[0])
    n_val = int(val_ratio * x.shape[0])
    n_test = x.shape[0] - n_train - n_val

    x_train, x_val, x_test = torch.split(x, [n_train, n_val, n_test])
    y_train, y_val, y_test = torch.split(y, [n_train, n_val, n_test])
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def compute_norm(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-feature mean/std *for all dimensions*.

    將極小方差直接設為 eps，確保沒有通道被跳過，避免 loss 權重失衡。
    """
    std = x.std(0)
    std_clamped = torch.clamp(std, min=eps)     # ← 每一維都真的被縮放
    mean = x.mean(0)
    return mean, std_clamped


def apply_norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply normalisation statistics to `x."""
    return (x - mean) / std


def undo_norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Revert normalisation applied to `x."""
    return x * std + mean

class MAAPELoss(nn.Module):
    """Mean Arctangent Absolute Percentage Error (MAAPE) loss."""
    def __init__(self, eps: float = 3e-4):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 避免除以 0
        denom = torch.clamp(target.abs(), min=self.eps)
        mape = (pred - target).abs() / denom
        # arctan 後取平均
        return torch.atan(mape).mean()


class OneMinusR2Loss(nn.Module):
    """Loss that minimises `1 - R^2 between pred and target."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_mean = target.mean(dim=0, keepdim=True)
        ss_tot = ((target - target_mean) ** 2).sum()
        ss_res = ((pred - target) ** 2).sum()
        r2 = 1.0 - ss_res / (ss_tot + self.eps)
        return 1.0 - r2


class MLP(nn.Module):
    """Two-layer MLP with optional BatchNorm and INT8 quantisation."""

    def __init__(
        self,
        input_dim: int = 23,          # 新的默認輸入維度：58−35=23
        hidden1: int = 64,
        hidden2: int = 32,
        output_dim: int = 12,
        dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        # ── QAT 量化節點 ──
        self.quant_in  = quant.QuantStub()
        self.dequant   = quant.DeQuantStub()

        self.fc1   = nn.Linear(input_dim, hidden1, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Linear(hidden1, hidden2, bias=True)
        self.relu2 = nn.ReLU()
        self.fc3   = nn.Linear(hidden2, output_dim, bias=True)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant_in(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x

    def fuse_model(self) -> None:
        # Linear+ReLU fusion (官方支援 pattern)
        quant.fuse_modules(
            self,
            [['fc1', 'relu1'],
             ['fc2', 'relu2']],
            inplace=True
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor
) -> float:
    """Evaluate model by un-normalising preds/targets and computing MSE on original scale."""
    model.eval()
    total_loss = 0.0
    mse_loss = nn.MSELoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        # forward
        pred = model(x)
        # un-normalise
        pred_orig   = undo_norm(pred,    y_mean.to(device), y_std.to(device))
        target_orig = undo_norm(y,       y_mean.to(device), y_std.to(device))
        # MSE on original scale
        loss = mse_loss(pred_orig, target_orig)
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def sample_val_prediction(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: torch.Tensor | None = None,
    y_std: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a single prediction/target pair from the validation set.

    If `y_mean and y_std are provided the values are unnormalised
    before being returned so they match the original scale of the dataset.
    """
    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    pred = model(x)
    pred = pred[0].cpu()
    y = y[0].cpu()
    if y_mean is not None and y_std is not None:
        pred = undo_norm(pred, y_mean, y_std)
        y = undo_norm(y, y_mean, y_std)
    return pred, y


def train_fp32(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    patience: int = 20,
    min_delta: float = 1e-4,
    weight_decay: float = 0.0,
    # Learning rate scheduler parameters
    scheduler_step_size: int = 30,
    scheduler_gamma: float = 0.1,
    criterion: nn.Module = nn.MSELoss(),   # 改用 MSE
) -> None:
    """Train the FP32 model using指定的 loss (MSE or MAAPE)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 使用 CosineAnnealingLR，T_max = 總 epoch 數
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=50, T_mult=2, eta_min=lr*0.01
    )

    best_loss = float("inf")
    wait = 0
    best_state = None

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        # --- training step & 计算 train_loss
        model.train()
        total_train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            # 累计训练损失（未乘 batch_size 归一化）
            total_train_loss += loss.item() * x.size(0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        train_loss_epoch = total_train_loss / len(train_loader.dataset)
        # 计算完 validation loss 后再触发 lr 调度

        # --- 验证 step
        val_loss = evaluate(model, val_loader, device, y_mean, y_std)
        # StepLR 不需要 metric，直接按 epoch 更新
        scheduler.step()

        train_losses.append(train_loss_epoch)
        val_losses.append(val_loss)

        if best_loss - val_loss > min_delta:
            best_loss = val_loss
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping FP32 at epoch {epoch+1}")
                break

        # 每 10 个 epoch 或在取得新最好时打印一次当前 lr & val loss
        current_lr = opt.param_groups[0]['lr']
        if (epoch + 1) % 10 == 0 or wait == 0:
            print(f"Epoch {epoch+1:3d}: val loss={val_loss:.6f}, lr={current_lr:.3e}")
            
            # 1) normalized predictions for magnitude check
            pred_norm, targ_norm = sample_val_prediction(model, val_loader, device, None, None)
            fmt_norm = [f'{n:+.1e}' for n in pred_norm.tolist()]
            print("      example pred_norm:", fmt_norm)
            # 2) un-normalized back to original scale
            pred_vec, target_vec = sample_val_prediction(model, val_loader, device, y_mean, y_std)
            formatted_pred   = [f'{n:+.1e}' for n in pred_vec.tolist()]
            formatted_target = [f'{n:+.1e}' for n in target_vec.tolist()]
            print("      example pred:     ", formatted_pred)
            print("      example target:   ", formatted_target)

    if best_state is not None:
        model.load_state_dict(best_state)

    # 绘制 train/val 曲线
    plt.figure()
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('FP32 Training and Validation Loss')
    plt.savefig(Path("loss_curve_fp32.png"))
    print("Saved FP32 loss curve to loss_curve_fp32.png")
    plt.close()

def train_qat(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    patience: int = 30,
    min_delta: float = 1e-4,
    weight_decay: float = 1e-5,
    scheduler_step_size: int = 30,
    scheduler_gamma: float = 0.1,
    criterion: nn.Module = nn.MSELoss(),
    freeze_observer_epoch: int | None = None,
    freeze_bn_epoch: int | None = None,
) -> None:
    """Quantisation aware training with standardised targets and指定的 loss."""
    model.train()

    # ────── 自動設定凍結時機（可覆寫） ──────
    if freeze_observer_epoch is None:
        freeze_observer_epoch = int(0.7 * epochs)
    if freeze_bn_epoch is None:
        freeze_bn_epoch = int(0.2 * epochs)
    model.quant = True

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=50, T_mult=2, eta_min=lr*0.01
    )

    best_loss = float("inf")
    wait = 0
    best_state = None

    for epoch in range(epochs):
        model.train()

        # ① 凍結 BN running mean/var（若模型有 BN 且函式存在）
        if epoch == freeze_bn_epoch and hasattr(quant, "freeze_bn_stats"):
            quant.freeze_bn_stats(model)
            print(f"[QAT] Freeze BN stats  @ epoch {epoch+1}")

        # ② 停止 observer -> 固定量化參數
        if epoch == freeze_observer_epoch:
            quant.disable_observer(model)
            print(f"[QAT] Disable observer @ epoch {epoch+1}")
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        # monitor validation loss 再触发 lr 调度

        val_loss = evaluate(model, val_loader, device, y_mean, y_std)
        scheduler.step()
        if best_loss - val_loss > min_delta:
            best_loss = val_loss
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping QAT at epoch {epoch+1}")
                break
        current_lr = opt.param_groups[0]['lr']
        print(f"[QAT] Epoch {epoch+1:3d}: val loss={val_loss:.6f}, lr={current_lr:.3e}")
        pred_vec, target_vec = sample_val_prediction(
            model, val_loader, device, y_mean, y_std
        )
            # 將 pred_vec 轉換為 list，並對其中每個數字格式化為科學記號且保留小數點後兩位
        formatted_pred = [f'{num:+.1e}' for num in pred_vec.tolist()]
        print("      example pred:  ", formatted_pred)

        # target_vec 通常是整數標籤，可能不需要格式化，此處保留原樣
        # 如果 target_vec 也是浮點數且需要格式化，可使用相同方法
        formatted_target = [f'{num:+.1e}' for num in target_vec.tolist()]
        print("      example target:", formatted_target)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP Kalman gain model")
    parser.add_argument("--csv", type=Path, default=Path("gru_training_data.csv"))
    parser.add_argument("--out", type=Path, default=Path("model_out"))
    parser.add_argument("--fp32-epochs", type=int, default=400)
    parser.add_argument("--qat-epochs", type=int, default=150)
    parser.add_argument("--hidden1", type=int, default=32)
    parser.add_argument("--hidden2", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fp32-lr", type=float, default=1e-3)
    parser.add_argument("--qat-lr", type=float, default=1e-4)
    parser.add_argument("--fp32-weight-decay", type=float, default=1e-5,
                        help="FP32 阶段的 weight decay，建议设为 1e-5")
    parser.add_argument("--qat-weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.05, help="use smaller dropout by default")
    parser.add_argument("--batchnorm", action="store_true", help="use BatchNorm layers")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    # 新增：FP32 阶梯学习率调度器的 step_size 和 gamma
    parser.add_argument("--scheduler-step-size", type=int, default=100,
                        help="StepLR 的 step_size，默认 30")
    parser.add_argument("--pretrain-epochs",   type=int, default=100,
                        help="FP32 阶段用 MSE 预热训练的 epoch 数")
    parser.add_argument("--scheduler-gamma", type=float, default=0.1,
                        help="StepLR 的 gamma，默认 0.1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--beta-huber", type=float, default=1e-3,
                        help="δ for SmoothL1Loss (Huber)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # —— 数据准备 & 模型初始化 ——
    x, y = load_dataset(args.csv)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(x, y)
    mean, std = compute_norm(x_train)
    x_train = apply_norm(x_train, mean, std)
    x_val   = apply_norm(x_val,   mean, std)
    x_test  = apply_norm(x_test,  mean, std)
    y_mean, y_std = compute_norm(y_train)
    y_train = apply_norm(y_train, y_mean, y_std)
    y_val   = apply_norm(y_val,   y_mean, y_std)
    y_test  = apply_norm(y_test,  y_mean, y_std)
    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        TensorDataset(x_val,   y_val),
        batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        TensorDataset(x_test,  y_test),
        batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=pin_mem,
    )


# --- ① 另外準備一份 **CPU 專用** 的 test_loader ---
#    - 不要 pin_memory
#    - 後面評估 INT8 時會用到
    test_loader_cpu = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    model = MLP(
        input_dim=x_train.shape[1],
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        output_dim=y_train.shape[1],
        dropout=args.dropout,
        batchnorm=False,         # 预热阶段不启用
    )
    model.apply(init_weights)
    model.to(device)

    # 單階段 MSE 訓練（驗證最快收斂）
    crit = nn.MSELoss()
    train_fp32(
        model, train_loader, val_loader,
        epochs=args.fp32_epochs, lr=args.fp32_lr,
        device=device, y_mean=y_mean, y_std=y_std,
        patience=args.patience, min_delta=args.min_delta,
        weight_decay=args.fp32_weight_decay,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        criterion=crit,
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_mean": mean,
            "x_std": std,
            "y_mean": y_mean,
            "y_std": y_std,
        },
        args.out / "model_fp32.pth",
    )

    torch.backends.quantized.engine = 'fbgemm'   # 選擇後端
    qat_model = copy.deepcopy(model)
    qat_model.train()   # ← 確保在 training mode，否則 prepare_qat 會噴錯
    qat_model.fuse_model()                       # 1. 先 fuse
    # 2. 指定 qconfig：per-channel 權重量化 & symmetric activation
    #    - activation: per-tensor symmetric (zero_point=0)
    #    - weight:     per-channel symmetric (zero_point=0)
    # 改用 HistogramObserver：對低幅度 activation 更友善
    activation_observer = quant.HistogramObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        bins=2048          # 若版本不支援 bins 也可直接拿掉
    )
    weight_observer = quant.PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,            # 通道軸通常是 0
        reduce_range=False
    )
    qat_model.qconfig = quant.QConfig(
        activation=activation_observer,
        weight=weight_observer
    )
    quant.prepare_qat(qat_model, inplace=True)   # 3. 插入 observer/fake-quant
    qat_model.to(device)
    train_qat(
        qat_model,
        train_loader,
        val_loader,
        args.qat_epochs,
        args.qat_lr,
        device,
        y_mean,
        y_std,
        patience=args.patience,
        min_delta=args.min_delta,
        weight_decay=args.qat_weight_decay,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        criterion=nn.MSELoss(),
    )

    # ───────── ② 量化前再做「完整資料」校正 ─────────
    #    重新開啟 observer，跑完整 train+val+test，然後再關閉
    qat_model.eval()
    quant.enable_observer(qat_model)
    with torch.no_grad():
        for loader in (train_loader, val_loader, test_loader_cpu):
            for x, _ in loader:
                qat_model(x.to("cpu"))
    quant.disable_observer(qat_model)

    # Convert to INT8
    quantized_model = quant.convert(qat_model.cpu(), inplace=False)

    # ───────── 新增：量化後用真驗證資料跑一次確認推論無誤 ─────────
    quantized_model.eval()
    with torch.no_grad():
        for x, _ in val_loader:
            quantized_model(x)

    scripted = torch.jit.script(quantized_model)
    scripted.save(str(args.out / "model_int8.pt"))

    # ③ FP32 評估（GPU）：反標準化後計算 MSE
    test_loss_fp32 = evaluate(model, test_loader, device, y_mean, y_std)

    # ④ INT8 評估（CPU）：反標準化後計算 MSE
    cpu_device = torch.device("cpu")
    test_loss_int8 = evaluate(
        model=quantized_model,
        loader=test_loader_cpu,
        device=cpu_device,
        y_mean=y_mean,
        y_std=y_std
    )

    with open(args.out / "metrics.json", "w") as f:
        json.dump({"fp32_test_mse": test_loss_fp32, "int8_test_mse": test_loss_int8}, f, indent=2)

    print(f"FP32 test loss: {test_loss_fp32:.6f}")
    print(f"INT8 test loss: {test_loss_int8:.6f}")



if __name__ == "__main__":
    main()