#!/usr/bin/env python3
import os
import torch
import numpy as np

def dump_human_readable(weights_path: str, out_dir: str):
    # 加载 .pth
    wb = torch.load(weights_path, map_location="cpu")
    os.makedirs(out_dir, exist_ok=True)

    for name, info in wb.items():
        w_int = info["int8_weight"]       # torch.int8
        b     = info["bias"]              # torch.float32
        scale = info["scale"]
        zp    = info["zero_point"]

        print(f"===== Layer: {name} =====")
        # 打印整数量化权重信息
        print(f"  int8_weight  shape: {tuple(w_int.shape)}")
        print(f"      min/max : {w_int.min().item()} / {w_int.max().item()}")
        # 打印浮点偏置信息
        print(f"  bias         shape: {tuple(b.shape)}")
        print(f"      min/max : {b.min().item():.6f} / {b.max().item():.6f}")
        # 打印量化参数
        print(f"  scale        : {scale}")
        print(f"  zero_point   : {zp}")
        print()

        # 导出整数量化权重
        # np.save(os.path.join(out_dir, f"{name}_int8_weight.npy"),
        #         w_int.detach().cpu().numpy())
        np.savetxt(os.path.join(out_dir, f"{name}_int8_weight.txt"),
                   w_int.detach().cpu().numpy().flatten(),
                   fmt="%d")

        # 导出偏置
        # np.save(os.path.join(out_dir, f"{name}_bias.npy"),
        #         b.detach().cpu().numpy())
        np.savetxt(os.path.join(out_dir, f"{name}_bias.txt"),
                   b.detach().cpu().numpy(),
                   fmt="%.6e")

    print(f"已导出所有层到目录：{out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                        default="./model_out/weights_bias.pth",
                        help="上一步生成的 .pth 文件")
    parser.add_argument("--out", type=str,
                        default="./model_out/readable",
                        help="导出目录")
    args = parser.parse_args()
    dump_human_readable(args.weights, args.out)
