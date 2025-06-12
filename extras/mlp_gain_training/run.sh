#python3 train_mlp_gain.py --seed 123 --loss maape --scheduler-gamma "0.1" --fp32-lr "1e-3" --eps-maape "3e-4" --batch-size 128
#Epoch  80: val loss=0.360304, lr=1.000e-06
#      example pred:   ['+8.0e-01', '+1.8e-05', '+1.4e-04', '+4.8e-01', '+1.8e-02', '+1.8e-05', '-9.8e-06', '-9.1e-03', '-1.1e-01', '+4.6e-05', '+0.0e+00', '+0.0e+00']
#      example target: ['+7.2e-02', '-1.6e-03', '-1.8e-02', '+8.3e-03', '+3.1e-03', '-7.4e-05', '-4.3e-05', '-8.2e-05', '-6.8e-02', '+1.8e-03', '+0.0e+00', '+0.0e+00']

python3 train_mlp_gain.py --seed 123 --loss maape --scheduler-gamma "0.1" --fp32-lr "1e-3" --eps-maape "3e-4" --batch-size 128

#python3 train_mlp_gain.py --seed 123 --loss maape --scheduler-gamma "0.1" --fp32-lr "1e-3" --eps-maape "3e-5" --batch-size 128
#Epoch  92: val loss=0.417474, lr=1.000e-08
#      example pred:   ['+5.9e-01', '-7.6e-03', '-1.8e-02', '+5.1e-01', '+6.2e-02', '-1.7e-03', '-5.7e-04', '-5.2e-03', '-3.5e+00', '+9.1e-02', '+3.3e-01', '+1.1e-01']
#      example target: ['+7.2e-02', '-1.6e-03', '-1.8e-02', '+8.3e-03', '+3.1e-03', '-7.4e-05', '-4.3e-05', '-8.2e-05', '-6.8e-02', '+1.8e-03', '+0.0e+00', '+0.0e+00']

#python3 train_mlp_gain.py --seed 123 --loss maape --scheduler-gamma "0.1" --fp32-lr "1e-3" --eps-maape "3e-4" --batch-size 128 --dropout 0.1
#Epoch  20: val loss=0.464222, lr=1.000e-05
#      example pred:   ['+8.0e-01', '+1.8e-05', '+1.4e-04', '+4.8e-01', '+1.8e-02', '+1.8e-05', '-9.8e-06', '-9.1e-03', '-1.1e-01', '+4.6e-05', '+0.0e+00', '+0.0e+00']
#      example target: ['+7.2e-02', '-1.6e-03', '-1.8e-02', '+8.3e-03', '+3.1e-03', '-7.4e-05', '-4.3e-05', '-8.2e-05', '-6.8e-02', '+1.8e-03', '+0.0e+00', '+0.0e+00']