# 📌 Official Implementation of "SBBTS: A Unified Schrödinger-Bass Framework for Synthetic Financial Time Series"

This repository contains the official implementation of the paper: [SBBTS: A Unified Schrödinger-Bass Framework for Synthetic Financial Time Series](https://arxiv.org/abs/2604.07159).


- **Authors**: Alexandre Alouadi, Grégoire Loeper, Célian Marsala, Othmane Mazhar, Huyên Pham
- **Contact**: alexandre.alouadi@gmail.com; huyen.pham@polytechnique.edu

---

## Abstract
We study the problem of generating synthetic time series that reproduce both marginal distributions and temporal dynamics, a central challenge in financial machine learning. Existing approaches typically fail to jointly model drift and stochastic volatility, as diffusion-based methods fix the volatility while martingale transport models ignore drift. We introduce the Schrödinger-Bass Bridge for Time Series (SBBTS), a unified framework that extends the Schrödinger-Bass formulation to multi-step time series. The method constructs a diffusion process that jointly calibrates drift and volatility and admits a tractable decomposition into conditional transport problems, enabling efficient learning. Numerical experiments on the Heston model demonstrate that SBBTS accurately recovers stochastic volatility and correlation parameters that prior SchrödingerBridge methods fail to capture. Applied to S&P 500 data, SBBTS-generated synthetic time series consistently improve downstream forecasting performance when used for data augmentation, yielding higher classification accuracy and Sharpe ratio compared to real-data-only training. These results show that SBBTS provides a practical and effective framework for realistic time series generation and data augmentation in financial applications. 

---

## Repository Structure

```text
SBBTS/
├── models/                  # Neural architectures (ScoreNN, encoder blocks, inverse maps)
├── training/                # Training loops, losses, early stopping
├── utils/                   # Data generation and utility helpers
├── data/                    # Input data artifacts (e.g., clusters)
├── data_augmentation/       # Feature/decomposition helpers for augmentation workflows
├── diffusion_dsbm.py        # SBBTS sampling / generation routines
├── run_heston.py            # End-to-end training + generation on Heston synthetic data
├── run_augmentation.py      # End-to-end training + generation for S&P500 cluster augmentation
├── augmentation_experiments.ipynb
└── requirements.txt
```

---

## Installation

### 1) Clone

```bash
git clone <your-fork-or-this-repo-url>
cd SBBTS
```

### 2) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick Start

### A) Heston experiment

Run the baseline Heston training + generation pipeline:

```bash
python run_heston.py
```

This script:

1. samples Heston paths,
2. converts prices to (scaled) log-returns,
3. trains the SBBTS score model,
4. generates synthetic trajectories via `generate_dsbm(...)`.

### B) S&P 500 augmentation experiment

```bash
python run_augmentation.py
```

This script:

1. loads a precomputed S&P500 cluster from `data/sp500_cluster_<id>.pkl`,
2. trains SBBTS on cluster time series,
3. samples synthetic paths for augmentation.

---

## Core Pipeline

At a high level, the workflow is:

1. **Prepare trajectories** in tensor format `(M, N, d)`.
2. **Normalize** (e.g., scale by empirical standard deviation).
3. **Train** `ScoreNN` with iterative SBBTS procedure (`training/training_sbbts_dsbm.py`).
4. **Infer initial latent state** `y_0` from trained transport relation.
5. **Generate** synthetic trajectories by Euler simulation + inverse transport (`diffusion_dsbm.py`).
6. **Rescale / postprocess** outputs back to desired domain (returns or prices).

---

## Main Hyperparameters

Typical knobs exposed in run scripts:

- `beta`: regularization / transport strength,
- `K`: number of outer SBBTS iterations,
- `N_pi`: Euler discretization steps per interval,
- `n_epochs`, `batch_size`, `lr`, `patience`, `delta`: training controls,
- `safe_t`: numerical safeguard near terminal time.

Start from defaults in `run_heston.py` and `run_augmentation.py`, then tune for your dataset size and horizon.

---

## Notes

- The scripts are designed for PyTorch and automatically select CUDA when available.
- For reproducible experiments, set explicit random seeds before data generation and training.
- If adapting to new assets/markets, keep preprocessing consistent between real and synthetic datasets.

---

## Citation

```bibtex
@misc{alouadi2026sbbtsunifiedschrodingerbassframework,
      title={SBBTS: A Unified Schr\"odinger-Bass Framework for Synthetic Financial Time Series}, 
      author={Alexandre Alouadi and Grégoire Loeper and Célian Marsala and Othmane Mazhar and Huyên Pham},
      year={2026},
      eprint={2604.07159},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.07159}, 
}
```

---

## Acknowledgments

This repository accompanies the SBBTS paper and is intended to support reproducible research on realistic synthetic financial time-series generation and augmentation. If you notice any errors or have suggestions for improvement, please feel free to reach out to us.
