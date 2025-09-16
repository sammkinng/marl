#!/usr/bin/env bash
set -e


# Run a baseline env test
python env.py


# Train a PPO agent for P0 (example)
python rl/train_ppo.py --config experiments/p0_train.yaml --seed 42