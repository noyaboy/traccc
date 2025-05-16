#!/usr/bin/env python
"""兩層 GRU-like MLP：Linear-tanh-Linear-tanh-Linear

此版本加入：
1. 自動將資料集拆分為 train / eval / test，並在訓練過程中回報 eval MSE，
   最後回報 test MSE（可視為 accuracy 指標）。
2. 透過指令列參數支援多樣訓練策略：batch size、學習率、最佳化器、
   與 learning-rate scheduler（Step / Cosine）。"""
import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

class KalmanNet(nn.Module):
    def __init__(self, hidden=32, d_out=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d_out),
        )
    def forward(self, x):
        return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",        required=True, help="由 generate_dataset.py 產生的 .npz")
    ap.add_argument("--epochs",      type=int, default=30)
    ap.add_argument("--hidden",      type=int, default=32)
    ap.add_argument("--batch",       type=int, default=8192)
    ap.add_argument("--lr",          type=float, default=1e-2)
    ap.add_argument("--optimizer",   type=str, choices=["adam", "sgd"], default="sgd")
    ap.add_argument("--scheduler",   type=str, choices=["step", "cosine"], default=None)
    ap.add_argument("--step-size",   type=int, default=10, help="StepLR 週期")
    ap.add_argument("--gamma",       type=float, default=0.1, help="StepLR 衰減率")
    ap.add_argument("--eval-split",  type=float, default=0.1)
    ap.add_argument("--test-split",  type=float, default=0.1)
    ap.add_argument("--epsilon",     type=float, default=0.1, help="誤差閾值 ε，用於 accuracy 評估")
    ap.add_argument("--out",         required=True)
    args = ap.parse_args()

    # 讀取資料並拆分
    npz = np.load(args.data)
    ds  = TensorDataset(torch.from_numpy(npz["x"]), torch.from_numpy(npz["y"]))

    n_total     = len(ds)
    n_test      = int(n_total * args.test_split)
    n_eval      = int(n_total * args.eval_split)
    n_train     = n_total - n_eval - n_test
    g           = torch.Generator().manual_seed(42)
    train_ds, eval_ds, test_ds = random_split(ds, [n_train, n_eval, n_test], generator=g)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    eval_dl  = DataLoader(eval_ds,  batch_size=args.batch, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    model = KalmanNet(args.hidden).float()

    # --- optimizer ---
    if args.optimizer == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- scheduler (optional) ---
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    else:
        scheduler = None

    lossf = nn.MSELoss()

    # --- training loop ---
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [train]", leave=False):
            pred = model(xb)
            loss = lossf(pred, yb)
            optim.zero_grad(); loss.backward(); optim.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            eval_losses = []
            eval_accs   = []
            for xb, yb in eval_dl:
                pred = model(xb)
                loss = lossf(pred, yb)
                eval_losses.append(loss.item())
                abs_err = (pred/yb - 1).abs()
                eval_accs.append((abs_err <= args.epsilon).float().mean().item())

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch+1:03d}: "
              f"train MSE={np.mean(train_losses):.6f}, "
              f"eval MSE={np.mean(eval_losses):.6f}, "
              f"eval Acc@ε={np.mean(eval_accs):.4f}")

    # --- final test evaluation ---
    model.eval()
    with torch.no_grad():
        test_losses = []
        test_accs   = []
        for xb, yb in test_dl:
            pred = model(xb)
            loss = lossf(pred, yb)
            test_losses.append(loss.item())
            abs_err = (pred/yb - 1).abs()
            test_accs.append((abs_err <= args.epsilon).float().mean().item())
    print(f"Test MSE : {np.mean(test_losses):.6f}, Test Acc@ε={np.mean(test_accs):.4f}")

    # 儲存權重以供 export_weights.py 使用
    torch.save(model.state_dict(), args.out)

if __name__ == "__main__":
    main()
