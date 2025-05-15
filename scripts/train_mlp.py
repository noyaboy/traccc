#!/usr/bin/env python
"""兩層 GRU-like MLP：Linear-tanh-Linear-tanh-Linear"""
import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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
    ap.add_argument("--data",    required=True)
    ap.add_argument("--epochs",  type=int, default=30)
    ap.add_argument("--hidden",  type=int, default=32)
    ap.add_argument("--out",     required=True)
    args = ap.parse_args()

    npz = np.load(args.data)
    ds  = TensorDataset(torch.from_numpy(npz["x"]), torch.from_numpy(npz["y"]))
    dl  = DataLoader(ds, batch_size=8192, shuffle=True)

    model = KalmanNet(args.hidden).float()
    optim = torch.optim.Adam(model.parameters(), 1e-3)
    lossf = nn.MSELoss()

    for epoch in range(args.epochs):
        for xb, yb in tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            pred = model(xb)
            loss = lossf(pred, yb)
            optim.zero_grad(); loss.backward(); optim.step()

    torch.save(model.state_dict(), args.out)

if __name__ == "__main__":
    main()
