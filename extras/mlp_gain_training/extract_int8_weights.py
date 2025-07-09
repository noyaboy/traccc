#!/usr/bin/env python3
import torch

def extract_int8_weights(model_path: str):
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    result = {}
    for name, mod in model.named_modules():
        # ←—— 只要这是一个 "_packed_params" 子模块就跳过
        if name.endswith("_packed_params"):
            continue

        if hasattr(mod, "_packed_params"):
            packed = mod._packed_params
            if hasattr(packed, "_packed_params"):
                packed = packed._packed_params

            w_q, b_float = torch.ops.quantized.linear_unpack(packed)
            w_int   = w_q.int_repr()
            w_float = w_q.dequantize()
            scale   = getattr(mod, "scale",     None)
            zero_pt = getattr(mod, "zero_point", None)

            result[name] = {
                "quantized_weight": w_q,
                "int8_weight":      w_int,
                "float_weight":     w_float,
                "bias":             b_float,
                "scale":            scale,
                "zero_point":       zero_pt,
            }

    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./model_out/model_int8.pt")
    parser.add_argument("--out",   type=str, default="./model_out/weights_bias.pth")
    args = parser.parse_args()

    wb = extract_int8_weights(args.model)
    torch.save(wb, args.out)
    print(f"已提取到 {len(wb)} 个量化子模块，结果保存在 `{args.out}`：")
    for module_name in wb:
        print("  ", module_name)
