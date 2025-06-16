# FP32 test loss:0.021942594072221884
# INT8 test loss:0.0932950983003395
python3 train_mlp_gain.py --seed 123 --fp32-lr 3e-3 --qat-lr 1e-4 --hidden1 48 --hidden2 32 \
  --fp32-epochs 1500 --qat-epochs 500 --batch-size 128 --patience 100 --min-delta 0 --beta-huber 1e-3
