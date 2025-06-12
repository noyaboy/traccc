#!/usr/bin/env python3

"""Train an MLP to approximate the Kalman gain.

The script expects a CSV file ``gru_training_data.csv`` containing rows of
58 input features followed by 12 target values. The first 58 columns encode
``predicted_vec`` (6), ``predicted_cov`` (36), ``H`` (12) and ``V`` (4).
The remaining 12 columns correspond to the flattened 6x2 Kalman gain matrix.

The training procedure consists of two stages:

1. FP32 pre-training to obtain good initial weights.
2. Quantisation aware training (QAT) to fine tune an INT8 friendly model.

The script saves ``model_fp32.pth`` and ``model_int8.pt`` in the output
folder.  Normalisation statistics for both the inputs and targets are stored
alongside the FP32 model and are applied during inference. Targets are
standardised during training and predictions are unnormalised back to the
original scale when reported.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple
import random
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def load_dataset(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load CSV dataset and split inputs/targets."""
    data = pd.read_csv(path, header=None, usecols=range(70))
    data = data.dropna(how='any')
    data = data.values.astype("float32")
    x = torch.tensor(data[:, :58])
    y = torch.tensor(data[:, 58:])
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


def compute_norm(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-feature mean and std for ``x``."""
    mean = x.mean(0)
    std = x.std(0)
    std[std == 0] = 1.0
    return mean, std


def apply_norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply normalisation statistics to ``x``."""
    return (x - mean) / std


def undo_norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Revert normalisation applied to ``x``."""
    return x * std + mean


class RMSELoss(nn.Module):
    """Root mean squared error loss."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(pred, target) + self.eps)


class R2Loss(nn.Module):
    """Loss based on the coefficient of determination (R^2).

    Minimises ``1 - R^2`` to maximise the quality of fit. The loss is computed
    over the entire batch.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - pred) ** 2)
        return ss_res / (ss_tot + self.eps)




class MLP(nn.Module):
    """Two-layer MLP with optional symmetric INT8 quantisation (scale=127)."""

    def __init__(self, input_dim=58, hidden1=32, hidden2=16, output_dim=12, quant=False, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1, bias=False)
        self.fc2 = nn.Linear(hidden1, hidden2, bias=False)
        self.fc3 = nn.Linear(hidden2, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.quant = quant

    @staticmethod
    def _fake_quant(x: torch.Tensor) -> torch.Tensor:
        return torch.fake_quantize_per_tensor_affine(
            x, scale=1.0 / 127.0, zero_point=0, quant_min=-128, quant_max=127
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant:
            x = self._fake_quant(x)
            w1 = self._fake_quant(self.fc1.weight)
        else:
            w1 = self.fc1.weight
        x = nn.functional.linear(x, w1)
        x = nn.functional.relu(x)
        x = self.dropout(x)

        if self.quant:
            x = self._fake_quant(x)
            w2 = self._fake_quant(self.fc2.weight)
        else:
            w2 = self.fc2.weight
        x = nn.functional.linear(x, w2)
        x = nn.functional.relu(x)
        x = self.dropout(x)

        if self.quant:
            x = self._fake_quant(x)
            w3 = self._fake_quant(self.fc3.weight)
        else:
            w3 = self.fc3.weight
        x = nn.functional.linear(x, w3)
        return x

    def fuse_model(self) -> None:
        pass  # kept for API compatibility


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        total_loss += criterion(pred, y).item() * x.size(0)
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

    If ``y_mean`` and ``y_std`` are provided the values are unnormalised
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
    min_delta: float = 0.0,
    weight_decay: float = 0.0,
) -> None:
    """Train the FP32 model using standardised targets and MSE loss."""
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)

    best_loss = float("inf")
    wait = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        scheduler.step()

        val_loss = evaluate(model, val_loader, criterion, device)
        if best_loss - val_loss > min_delta:
            best_loss = val_loss
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping FP32 at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0 or wait == 0:
            print(f"Epoch {epoch+1:3d}: val loss={val_loss:.6f}")
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
    min_delta: float = 0.0,
    weight_decay: float = 1e-5,
) -> None:
    """Quantisation aware training with standardised targets and MSE loss."""
    model.train()
    model.quant = True

    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.1)

    best_loss = float("inf")
    wait = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        scheduler.step()

        val_loss = evaluate(model, val_loader, criterion, device)
        if best_loss - val_loss > min_delta:
            best_loss = val_loss
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping QAT at epoch {epoch+1}")
                break

        print(f"[QAT] Epoch {epoch+1:3d}: val loss={val_loss:.6f}")
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
    parser.add_argument("--fp32-epochs", type=int, default=100)
    parser.add_argument("--qat-epochs", type=int, default=150)
    parser.add_argument("--hidden1", type=int, default=32)
    parser.add_argument("--hidden2", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fp32-lr", type=float, default=1e-3)
    parser.add_argument("--qat-lr", type=float, default=1e-4)
    parser.add_argument("--fp32-weight-decay", type=float, default=0.0)
    parser.add_argument("--qat-weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    x, y = load_dataset(args.csv)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(x, y)

    mean, std = compute_norm(x_train)
    x_train = apply_norm(x_train, mean, std)
    x_val = apply_norm(x_val, mean, std)
    x_test = apply_norm(x_test, mean, std)

    y_mean, y_std = compute_norm(y_train)
    y_train = apply_norm(y_train, y_mean, y_std)
    y_val = apply_norm(y_val, y_mean, y_std)
    y_test = apply_norm(y_test, y_mean, y_std)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size)

    args.out.mkdir(parents=True, exist_ok=True)

    model = MLP(hidden1=args.hidden1, hidden2=args.hidden2, dropout=args.dropout)
    model.to(device)
    train_fp32(
        model,
        train_loader,
        val_loader,
        args.fp32_epochs,
        args.fp32_lr,
        device,
        y_mean,
        y_std,
        patience=args.patience,
        min_delta=args.min_delta,
        weight_decay=args.fp32_weight_decay,
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

    qat_model = MLP(hidden1=args.hidden1, hidden2=args.hidden2, dropout=args.dropout)
    qat_model.load_state_dict(model.state_dict())
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
    )

    scripted = torch.jit.script(qat_model)
    scripted.save(str(args.out / "model_int8.pt"))

    test_loss_fp32 = evaluate(model, test_loader, nn.MSELoss(), device)
    test_loss_int8 = evaluate(qat_model, test_loader, nn.MSELoss(), device)

    with open(args.out / "metrics.json", "w") as f:
        json.dump({"fp32_test_mse": test_loss_fp32, "int8_test_mse": test_loss_int8}, f, indent=2)

    print(f"FP32 test loss:{test_loss_fp32}")
    print(f"INT8 test loss:{test_loss_int8}")



if __name__ == "__main__":
    main()
