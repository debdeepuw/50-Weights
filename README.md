# Wasserstein Constrained Empirical Likelihood (wcETEL)

## Overview

This repository implements the Wasserstein-constrained exponentially tilted empirical likelihood (wcETEL) method, following:

> Chakraborty, A., Bhattacharya, A., & Pati, D. (2023). Robust probabilistic inference via a constrained transport metric. *arXiv preprint arXiv:2303.10085*.

The wcETEL method performs robust inference by combining:
- Empirical likelihood (EL)
- Wasserstein distance constraints

The Wasserstein penalty allows downweighting of contaminated data points and makes inference more robust under model misspecification.

This repository provides fully reproducible code for simulating datasets, computing wcETEL weights, and visualizing the results under various contamination levels.

---

## Data Generating Models

We simulate 50 datasets for each of three contamination settings:

| Setting | Main signal | Contamination |
|---------|--------------|----------------|
| Clean   | 300 samples from Beta(2,2) | None |
| Mild    | 300 samples, with 14 contaminated points (7 Beta(1,100) and 7 Beta(100,1)) | Mild contamination |
| Heavy   | 300 samples, with 60 contaminated points (30 Beta(1,100) and 30 Beta(100,1)) | Heavy contamination |

### Data generation code

```python
def generate_datasets(mode="heavy", N=300, n_datasets=50):
    datasets = []
    for i in range(n_datasets):
        np.random.seed(1000 + i)
        if mode == "clean":
            sig = np.random.beta(2, 2, size=N)
            X = np.sort(sig)
        elif mode == "mild":
            sig = np.random.beta(2, 2, size=N - 14)
            noi_1 = np.random.beta(1, 100, size=7)
            noi_2 = np.random.beta(100, 1, size=7)
            X = np.sort(np.concatenate([sig, noi_1, noi_2]))
        elif mode == "heavy":
            sig = np.random.beta(2, 2, size=N - 60)
            noi_1 = np.random.beta(1, 100, size=30)
            noi_2 = np.random.beta(100, 1, size=30)
            X = np.sort(np.concatenate([sig, noi_1, noi_2]))
        datasets.append(X)
    return datasets
```

### Example plot (heavy contamination)

![Heavy contamination plot](plots/heavy_weights_overlay.png)

---

## Repository Structure

| File | Description |
|------|-------------|
| `data_generator.py` | Generate simulated datasets under different contamination settings |
| `run_wcETEL_analysis_module.py` | Implements wcETEL iterative weight updates using semi-discrete optimal transport |
| `optimal_transport_1d.py` | Power diagram & semi-discrete OT computations |
| `parallel_runner.py` | Parallel execution of wcETEL weight computation across datasets |
| `scripts/run_clean_mild_heavy.py` | Automates generating datasets and running analysis for all 3 settings |
| `scripts/plot_weights_overlay.py` | Generate overlay plots of weights for each contamination setting |
| `wcETEL_results/` | Stores output weights and intermediate results |
| `plots/` | Stores generated weight overlay plots for clean, mild, and heavy contamination |

---



## Running the Code

### 1. Run the full pipeline

The entire pipeline can be run directly from the `parallel.ipynb` notebook.

The notebook will:
- Launch 3 contamination settings in parallel.
- Compute wcETEL weights using `parallel_runner.py`.
- Generate and save all weight overlay plots.
- Display all plots directly inside the notebook.

### 2. Manual execution via command line (optional)

If you prefer to run each step manually:

#### Compute weights

```bash
python scripts/run_clean_mild_heavy.py
```
#### Generate plots

```bash
python scripts/plot_weights_overlay.py
```

## Method Summary

The wcETEL method solves the following optimization problem:

**w̃(θ) = argmin₍w₎ { D(Qw ∥ Pₙ) + λ W₂²(Qw, Fθ) }**

- $Q_w$ is a discrete distribution over the observed data points.
- $P_n$ is the empirical distribution.
- The Wasserstein distance $W_2$ is computed using semi-discrete optimal transport with power diagrams.

The iterative procedure updates weights using a fixed-point algorithm that combines empirical likelihood and optimal transport.

