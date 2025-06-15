# FP32 val loss=0.041657
python3 train_mlp_gain.py --seed 123 --fp32-lr 3e-3 --qat-lr 1e-4 --hidden1 48 --hidden2 16 \
  --fp32-epochs 800 --qat-epochs 500 --batch-size 128 --patience 100 --min-delta 0 --beta-huber 1e-3
