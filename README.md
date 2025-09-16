# MARL-QKD Repository Skeleton

This repository skeleton helps you implement P0 → P2 for the MARL in QKD project. It includes a minimal simulator (`env.py`) (the one we just tested), RL training scaffolding, experiment configs, plotting and reproducibility notes.

---

## File / Directory structure

```
marl-qkd/                     # root
├── README.md                  # this file
├── requirements.txt           # pip requirements
├── .gitignore
├── LICENSE
├── env.py                     # BB84 + decoy environment (starter)
├── run.sh                     # convenience script to run experiments

├── rl/                        # RL training code
│   ├── train_ppo.py           # PPO training script (Stable-Baselines3 or custom)
│   ├── agent.py               # policy network definitions & wrappers
│   ├── replay.py              # optional: replay / buffer utilities
│   └── policies/              # saved policies & policy pool management
│       └── README.md

├── sims/                      # simulation utilities & extensions
│   ├── time_shift_eve.py      # Eve: time-shift attack implementation
│   ├── pns_eve.py             # Eve: PNS attack implementation (P1)
│   ├── noise_eve.py           # Eve: noise injection (P2)
│   └── ldpc_model.py          # simplified LDPC / reconciliation model

├── experiments/               # manifest of experiment configs & scripts
│   ├── p0_baseline.yaml       # experiment config for P0 baseline
│   ├── p0_train.yaml          # training hyperparams for P0
│   └── run_experiments.sh     # orchestrates experiment runs

├── analysis/                  # plotting & result analysis
│   ├── plot_learning_curves.py
│   ├── plot_skr_vs_loss.py
│   └── summarize_results.ipynb

├── notebooks/                 # quick interactive experiments
│   └── sweep_skr_vs_eta.ipynb

└── docs/
    ├── experiment_plan.md
    └── paper_figures/         # generated figs for paper
```

---

## README.md (quick guide)

```
# MARL-QKD

This repository contains a minimal BB84 + decoy-state environment and an RL training scaffold to implement MARL defenses and attacks for QKD.

## Quickstart

1. Create a virtualenv: `python -m venv venv && source venv/bin/activate`
2. Install requirements: `pip install -r requirements.txt`
3. Run a baseline simulation:
   `python env.py`

4. Train a PPO policy (example):
   `python rl/train_ppo.py --config experiments/p0_train.yaml`

5. Run evaluation and produce plots:
   `python analysis/plot_skr_vs_loss.py --results results/p0`

## Project layout
See the repository structure in the root `README.md`.

## Reproducibility
- Seeds: pass `--seed` to scripts and record it in results.
- Configs: keep YAML/JSON configs in `experiments/` and keep a `results/` folder with timestamps.
```

---

## requirements.txt

```
numpy
scipy
matplotlib
pandas
pyyaml
tqdm
torch
stable-baselines3  # optional; or use RLlib/PPO custom
wandb               # optional
```

---

## Key starter scripts

### `rl/train_ppo.py` (summary)

* Loads env (env.BB84Env) and wraps as gym-like env.
* Instantiates PPO agents for Alice, Bob, and Eve (Eve can be a fixed scripted policy initially).
* Training loop supports alternating updates or simultaneous updates.
* Checkpoint saving & logging (TensorBoard / W\&B).

\_Create `rl/train_ppo.py` with CLI args: `--config`, `--seed`, `--checkpoint-dir`, `--gpu`

### `sims/time_shift_eve.py`

* Implements `TimeShiftEve` class that exposes configurable `attack_fraction`, `time_shift_amount`.
* Provides an `act()` method compatible with `env.step({'Eve': ...})`.

### `analysis/plot_skr_vs_loss.py`

* Loads results CSVs and plots SKR vs channel transmittance (eta) with error bars.
* Plots learning curves from logs (SKR vs training steps).

---

## Example `run.sh`

```bash
#!/usr/bin/env bash
set -e

# Run a baseline env test
python env.py

# Train a PPO agent for P0 (example)
python rl/train_ppo.py --config experiments/p0_train.yaml --seed 42
```

Make it executable: `chmod +x run.sh`

---

## experiments/p0\_train.yaml (example config)

```yaml
env:
  pulses_per_episode: 5000
  mu_signal: 0.5
  mu_decoy: 0.1
  mu_vac: 0.0
  p_signal: 0.6
  p_decoy: 0.3
  p_vac: 0.1
  eta: 0.1
  dark_count: 1e-6
  det_eff: 0.6
  basis_prob: 0.5

train:
  algo: ppo
  total_timesteps: 2_000_000
  learning_rate: 3e-4
  gamma: 0.99
  clip_range: 0.2
  batch_size: 2048
  n_epochs: 10

logging:

  logdir: results/p0
  save_interval: 10000
  use_wandb: false

opponent:
  type: fixed  # or 'learned' to train Eve
```

---

## Repro tips

* Save both code & config with each result (e.g., tar the commit hash + config file).
* Provide `evaluate.py` that loads checkpoints and runs standardized evaluation across seeds.

---

