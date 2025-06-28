#!/usr/bin/env python3
import torch
from pathlib import Path
import sys

# Determine model path: script dir/model_int8.pt or model_out/model_int8.pt
cwd = Path(__file__).parent
candidates = [cwd / "model_int8.pt", cwd / "model_out" / "model_int8.pt"]
for p in candidates:
    if p.exists():
        model_path = p
        break
else:
    print("Error: cannot find model_int8.pt in cwd or model_out.")
    sys.exit(1)

# Load quantized TorchScript model
model = torch.jit.load(str(model_path))

# Prepare output directory and file
output_dir = cwd / "model_out" / "readable"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "qparams.txt"

lines = []

# 1) Activation quant params
lines.append("=== Activation quant params ===")
for name, mod in model.named_modules():
    if hasattr(mod, 'scale') and hasattr(mod, 'zero_point'):
        lines.append(f"{name}: scale={float(mod.scale):e}, zero_point={int(mod.zero_point)}")

# 2) Weight quant params via modules
lines.append("")
lines.append("=== Weight quant params ===")
for name, mod in model.named_modules():
    # try to get quantized weight tensor
    try:
        w_q = mod.weight()
    except Exception:
        continue
    if not hasattr(w_q, 'is_quantized') or not w_q.is_quantized:
        continue

    # distinguish per-channel vs per-tensor
    if hasattr(w_q, 'q_per_channel_scales'):
        scales = w_q.q_per_channel_scales().tolist()
        zeros = w_q.q_per_channel_zero_points().tolist()
        lines.append(f"{name}: (per-channel)")
        lines.append("  scales      = " + str([f"{s:e}" for s in scales]))
        lines.append("  zero_points = " + str(zeros))
    else:
        lines.append(f"{name}: (per-tensor) scale={w_q.q_scale():e}, zero_point={w_q.q_zero_point()}")

# Write out
output_text = "\n".join(lines)
print(output_text)
with open(output_file, 'w') as f:
    f.write(output_text)
print(f"\nQuantization parameters saved to {output_file}")
