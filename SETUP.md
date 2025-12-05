# Project Setup Guide

This document provides step-by-step instructions for setting up the environment required to run all components of this project:

- `data.py` — supervised learning data preprocessing and feature engineering
- `model.py` — supervised kicker-agnostic probability model
- `rl.ipynb` — reinforcement learning field goal environment, DQN agent, and evaluation

These instructions assume Python 3.9+ is installed.

---

## 1. Clone the Repository

`git clone https://github.com/pcd15/field-goal-ml.git
cd field-goal-ml`

## 2. Create a Virtual Environment

`python3 -m venv .venv`

## 3. Activate the Virtual Environment

macOS / Linux: `source .venv/bin/activate`

## 4. Install All Dependencies

All required Python packages are listed in requirements.txt.

`pip install --upgrade pip
pip install -r requirements.txt`

## 5. Register the Virtual Environment as a Jupyter Kernel


`python -m ipykernel install --user --name field-goal-ml --display-name "field-goal-ml"`

You will later select this kernel inside JupyterLab.

## 6. Launch JupyterLab

`jupyter lab`

Then, open rl.ipynb and select the kernel named field-goal-ml

Run cells top-to-bottom to:
- build the custom Gymnasium environment
- train the DQN reinforcement learning agent
- generate evaluation plots (automatically saved to ../plots/)

## 7. Running Supervised Learning Code

`python data.py`
`python model.py`
