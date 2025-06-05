# Deep Reinforcement Learning for Financial Asset Recommendation

## Overview

This project applies Deep Reinforcement Learning (DQN, DDQN, A2C) to personalized financial asset recommendation. The goal is to suggest assets that balance return, risk, and diversification. It uses the FAR-Trans dataset and a custom financial environment. Configuration is managed via Hydra.

---

## Project Structure

```

financial\_rl\_project/
├── conf/                 # Hydra configs (agent, env, data, training)
├── data/
│   └── raw/              # Raw CSVs or datasets
├── Documents/            # Pdf Report + slides
├── notebooks/            # Jupyter notebooks
├── src/
│   ├── agent/            # DRL agents & neural networks
│   ├── data\_management/  # Data loaders and generators
│   ├── environment/      # FinancialRecEnv environment
│   ├── evaluation/       # Evaluation and metrics
│   ├── heuristics/       # Baselines: MVO, RP, Equal Weight
│   ├── training/         # Training loops & Optuna tuning
│   ├── utils/            # Utilities: seeding, plotting, etc.
│   └── run.py            # Main Hydra entry point
├── tests/                # Unit/integration tests
├── requirements.txt      # Dependencies
├── .gitignore
├── LICENSE
└── README.md

````

---

## Setup

### 1. Create Environment (Conda)
```bash
conda create -n drl_finance python=3.10
conda activate drl_finance
````

### 2. Install Dependencies

```bash
git clone <your-repo-url>
cd financial_rl_project
pip install -r requirements.txt
```

Optional dependencies like `cvxpy` (for MVO) and `plotly` (for Optuna plots) can be installed separately.

### 3. (Optional) Editable Install

```bash
pip install -e .
```

---

## Running the Project

### Default Run

```bash
python src/run.py
```

### Specify Agent

```bash
python src/run.py agent=a2c_default
python src/run.py agent=ddqn_default
```

(Ensure corresponding YAMLs exist in `conf/agent/`)

### Other Config Overrides

Hydra allows dynamic overrides:

```bash
python src/run.py  training.num_episodes=100
```


### Dislaimer

Part of this code were commented, factorized and/or completed by an AI assissant.

### Funding

this work is supported by the MSCA Digital ( Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or Horizon Europe: Marie Skłodowska-Curie Actions. Neither the European Union nor the granting authority can be held responsible for them. This project has received funding from the Horizon Europe research and innovation programme under the Marie Skłodowska-Curie )