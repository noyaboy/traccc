#!/usr/bin/env python
"""將 PyTorch 權重導出成 C++ constexpr array（浮點 + INT8）"""
import argparse, torch, numpy as np, textwrap, pathlib

def flatten_weights(state):
    # 與 C++ rnd() 索引對應
    hidden, inp = state["net.0.weight"].shape  # (32,6)
    k_rnd = []

    # --- W0 ---
    for i in range(hidden):
        for j in range(inp):
            k_rnd.append(state["net.0.weight"][i, j].item())
    # --- B0 ---
    k_rnd.extend(state["net.0.bias"].tolist())

    # --- W1 ---
    for i in range(hidden):
        for j in range(hidden):
            k_rnd.append(state["net.2.weight"][i, j].item() )
    # padding 到 5000
    k_rnd = k_rnd + [0.0]*(5000-len(k_rnd))

    # --- W1 offset 5000 ---
    w1_off = len(k_rnd)
    for i in range(hidden):
        for j in range(hidden):
            k_rnd.append(state["net.2.weight"][i, j].item())
    # --- B1 offset 20000 ---
    while len(k_rnd) < 20000: k_rnd.append(0.0)
    k_rnd.extend(state["net.2.bias"].tolist())

    # --- Dense B 30000 ---
    while len(k_rnd) < 30000: k_rnd.append(0.0)
    k_rnd.extend(state["net.4.bias"].tolist())

    # --- Dense W 40000 ---
    while len(k_rnd) < 40000: k_rnd.append(0.0)
    out, _ = state["net.4.weight"].shape  # 12×32
    for o in range(out):
        for j in range(hidden):
            k_rnd.append(state["net.4.weight"][o, j].item())

    return np.array(k_rnd, np.float32)

def write_float_hpp(arr, path):
    path = pathlib.Path(path)
    with open(path, "w") as f:
        f.write(textwrap.dedent(f"""\
            #pragma once
            #include <array>
            namespace traccc::fitting::trained {{
            constexpr std::array<float,{len(arr)}> kRnd = {{
            """))
        for v in arr:
            f.write(f"  {v:.8e}f,\n")
        f.write("};\n}\n")

def write_int8_hpp(arr, path):
    q = np.clip(np.round(arr*127), -128, 127).astype(np.int8)
    path = pathlib.Path(path)
    with open(path, "w") as f:
        f.write(textwrap.dedent(f"""\
            #pragma once
            #include <array>
            namespace traccc::fitting::trained {{
            constexpr std::array<std::int8_t,{len(q)}> kRndInt8 = {{
            """))
        for v in q:
            f.write(f"  {int(v)},\n")
        f.write("};\n}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--float-hpp",  required=True)
    ap.add_argument("--int8-hpp",   required=True)
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    arr   = flatten_weights(state)
    write_float_hpp(arr, args.float_hpp)
    write_int8_hpp(arr,  args.int8_hpp)

if __name__ == "__main__":
    main()
