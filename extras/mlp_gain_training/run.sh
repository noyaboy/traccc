# python3 train_mlp_gain.py --seed 123 --loss maape --scheduler-gamma "0.1" --fp32-lr "1e-3" --eps-maape "3e-4"
# Epoch  90: val loss=0.384181, lr=1.000e-08
#       example pred:   ['+9.0e-01', '-1.0e-03', '-4.6e-04', '+2.6e-01', '+4.3e-03', '-3.8e-06', '-4.4e-05', '-5.0e-04', '-1.8e-02', '+5.7e-06', '-3.2e-11', '+1.9e-10']
#       example target: ['+8.0e-01', '-2.4e-03', '-3.8e-02', '+4.5e-01', '+4.6e-03', '-1.3e-05', '-7.2e-05', '-4.2e-04', '-1.2e-02', '+6.3e-05', '+0.0e+00', '+0.0e+00']

# python3 train_mlp_gain.py --seed 123 --loss maape --scheduler-gamma "0.1" --fp32-lr "1e-3" --eps-maape "3e-5"
# Epoch 130: val loss=0.454068, lr=1.000e-08
#       example pred:   ['+8.8e-01', '-1.2e-03', '-6.6e-02', '+2.4e-01', '+6.0e-03', '-7.4e-06', '-4.8e-05', '-2.8e-04', '-1.9e-02', '+7.7e-06', '-2.5e-10', '-1.2e-01']
#       example target: ['+8.0e-01', '-2.4e-03', '-3.8e-02', '+4.5e-01', '+4.6e-03', '-1.3e-05', '-7.2e-05', '-4.2e-04', '-1.2e-02', '+6.3e-05', '+0.0e+00', '+0.0e+00']

python3 train_mlp_gain.py --seed 123 --loss maape --scheduler-gamma "0.1" --fp32-lr "1e-3" --eps-maape "1e-5"