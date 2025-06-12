# Kalman Gain MLP Training

This directory contains a Python script to train a feed-forward neural network
that predicts the Kalman gain matrix. Training data is expected in the
`gru_training_data.csv` format produced by the tracking code.

The workflow implements two stages:

1. **FP32 training** – learn a baseline model in full precision.
2. **Quantisation aware training** – fine tune the model to produce
   an INT8 friendly network.

The resulting model architecture matches the implementation in
`kalman_int8_gru_gain_predictor.hpp` (hidden sizes 32 and 16).
Training minimises ``1 - R^2`` so as to maximise the coefficient of
determination between the predicted and true Kalman gain. Run the script with
`python3 train_mlp_gain.py --help` for usage information.

## Example run

The script `run.sh` contains a set of example commands.  The last command in
that file trains a slightly larger network with BatchNorm and dropout and runs
for more epochs.  It is tuned to bring the FP32 validation loss down to around
``0.05`` while keeping the INT8 model close.  Adjust the hyperparameters if your
dataset behaves differently.
