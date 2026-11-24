# Analog Entanglement Distillation through Hamiltonian Scrambling

Simulation code and figure scripts for [*Analog entanglement distillation through Hamiltonian scrambling*](https://arxiv.org/abs/2603.10843v2) by Zitai Xu, Guoding Liu, and Xingjian Zhang ([arXiv:2603.10843v2](https://arxiv.org/abs/2603.10843v2)).

The scripts in this repository simulate continuous-time entanglement distillation under native many-body Hamiltonians (Ising, trapped-ion, Rydberg-atom, and diagonal models), evaluate finite-time dynamics and non-Pauli amplitude-damping noise, and compute error thresholds for quantum key distribution (QKD) and quantum repeaters.

## Step-by-Step Reproduction Guide

All commands below should be run from the repository root (`AEDP/`). Generated figures are saved as vector graphics (`.svg`) in `outputs/`.

### Step 1: Install Dependencies

Use Python 3.10 or newer:

```bash
pip install -r requirements.txt
```

To verify the installation and check the sampling kernels against exact Pauli enumeration on small systems ($n = 2, 3$):

```bash
python -m unittest discover -s tests
```

*(Optional)* On a headless server without a display, prefix plotting commands with `MPLBACKEND=Agg`. All scripts automatically use a CUDA GPU when available and fall back to CPU otherwise.

### Step 2: Generate the Figures

| Paper Figure | Topic | Command | Output (`outputs/`) |
| :--- | :--- | :--- | :--- |
| **Fig. 1** | Protocol schematic | Compiled from `paper/tikz/hedp-overview.tex` | *(LaTeX/TikZ source)* |
| **Fig. 2** | Fidelity & yield vs. measured pairs $m$ | `python simulations/variance_reduction_sampling.py --n 10 --measured 2 3 4 5 6 --samples 3600` | `fym_variance_reduced_n10_samples3600.svg` |
| **Fig. 3** | Fidelity & yield vs. depolarizing noise $p$ | `python simulations/fyp.py` | `Fyp.svg` |
| **Fig. 4** | Finite-time evolution ($T \in [0.05, 30]$) | `python simulations/finite_time.py` | `finite_time.svg` |
| **Fig. 5** | Amplitude damping with & without Pauli twirling | `python simulations/noise_comparison.py` | `amplitude_damping.svg` |
| **Fig. 6(a)** | Asymptotic tolerable error rate vs. $m/n$ | `python threshold/sim.py` | `tolerable.svg` |
| **Fig. 6(b)** | Finite-size tolerable error rate ($n, m \le 100$) | `python threshold/grid.py` | `grid.svg` |
| **Fig. 7(a)** | Maximum QKD transmission distance | `python applications/qkd_distance.py` | `qkd_distance.svg` |
| **Fig. 7(b)** | Repeater fidelity vs. distance | `python applications/repeater.py` | `repeater.svg` |

#### 1. Fidelity and yield vs. number of measured pairs (Figure 2)

At $n = 10$ ($p = 0.2, m = 2, \dots, 6$), `variance_reduction_sampling.py` evaluates low-weight Pauli errors ($w \le 2$) exactly and uses stratified sampling for higher weights ($w \ge 3$):

```bash
python simulations/variance_reduction_sampling.py --n 10 --measured 2 3 4 5 6 --samples 3600
```

For a fast check before running the 10-qubit sweep, use $n = 6$ (runs in a few seconds on CPU):

```bash
python simulations/variance_reduction_sampling.py --n 6 --measured 2 3 4 --samples 3600
```

An ordinary unstratified Monte Carlo sampler is also available via `python simulations/sampled_edp.py` with the same `--n`, `--measured`, `--p`, `--samples`, and `--device` flags.

#### 2. Depolarizing noise sweep (Figure 3)

Runs exact Pauli enumeration for $n = 6, m = 3$ across 10 depolarizing noise strengths $p \in [0.05, 0.30]$:

```bash
python simulations/fyp.py
```

#### 3. Finite-time evolution (Figure 4)

Simulates explicit 10-qubit ($n = 5, m = 3, p = 0.2$) density-matrix time evolution for trapped-ion and Rydberg Hamiltonians across 50 time points $T \in [0.05, 30]$:

```bash
python simulations/finite_time.py
```

#### 4. Non-Pauli amplitude-damping noise (Figure 5)

Compares distillation fidelity with and without Pauli twirling for $n = 5, m = 3, p = 0.2$ and damping strength $\gamma \in [0, 0.20]$:

```bash
python simulations/noise_comparison.py
```

#### 5. Error thresholds and network applications (Figures 6 & 7)

These four scripts use analytical rate formulas and run in a few seconds on CPU:

```bash
python threshold/sim.py
python threshold/grid.py
python applications/qkd_distance.py
python applications/repeater.py
```

## Repository Layout

```text
AEDP/
├── simulations/
│   ├── torch_edp.py                    # Hamiltonian builders and detection kernels
│   ├── sample_utils.py                 # Batched Pauli row-permutation evaluator
│   ├── variance_reduction_sampling.py  # Weight-stratified sampler (Figure 2)
│   ├── sampled_edp.py                  # Standard i.i.d. Pauli sampler
│   ├── fyp.py                          # Depolarizing noise sweep (Figure 3)
│   ├── finite_time.py                  # Finite-time evolution (Figure 4)
│   ├── noise_comparison.py             # Amplitude damping comparison (Figure 5)
│   ├── amplitude_damping.py            # Twirled amplitude-damping routines
│   ├── infinite_time.py                # Untwirled long-time projection routines
│   ├── recurrence.py                   # 1- and 2-round recurrence baselines
│   └── utils.py                        # Single-sample Pauli generator
├── threshold/
│   ├── sim.py                          # Asymptotic tolerable error rate (Figure 6a)
│   └── grid.py                         # Finite-size tolerable error rate (Figure 6b)
├── applications/
│   ├── qkd_utils.py                    # Fiber channel parameters and root solver
│   ├── qkd_distance.py                 # QKD maximum distance (Figure 7a)
│   └── repeater.py                     # Quantum repeater fidelity (Figure 7b)
├── outputs/                            # Generated SVG figures
├── paper/                              # Manuscript TeX source and reference figures
├── tests/                              # Unit tests for sampling and detection kernels
└── requirements.txt                    # Python package list
```

## Reference

Paper link: [arXiv:2603.10843v2](https://arxiv.org/abs/2603.10843v2)

```bibtex
@article{xu2026analog,
  title   = {Analog entanglement distillation through Hamiltonian scrambling},
  author  = {Xu, Zitai and Liu, Guoding and Zhang, Xingjian},
  journal = {arXiv preprint arXiv:2603.10843},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.10843v2}
}
```