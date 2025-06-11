# Kalman Gain MLP Training

This directory contains a Python script to train a feed-forward neural network
that predicts the Kalman gain matrix. Training data is expected in the
`gru_training_data.csv` format produced by the tracking code.

The workflow implements two stages:

1. **FP32 training** – learn a baseline model in full precision.
2. **Quantisation aware training** – fine tune the model to produce
   an INT8 friendly network.

Run the script with `python3 train_mlp_gain.py --help` for usage
information.
