
# FP32 test loss: 0.000033
# INT8 test loss: 0.000057
python3 train_mlp_gain.py --seed 123 --fp32-lr 3e-3 --qat-lr 1e-4 --hidden1 32 --hidden2 64 \
  --fp32-epochs 200 --qat-epochs 100 --batch-size 128 --patience 100 --min-delta 1e-5 --beta-huber 1e-3

