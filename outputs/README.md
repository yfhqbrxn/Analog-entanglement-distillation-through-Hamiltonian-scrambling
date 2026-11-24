# Generated Figures

Running the scripts from the repository root saves the vector figures (`.svg`) here:

| Figure | Command | Output File |
| :--- | :--- | :--- |
| **Fig. 2** | `python simulations/variance_reduction_sampling.py --n 10 --measured 2 3 4 5 6 --samples 3600` | `fym_variance_reduced_n10_samples3600.svg` |
| **Fig. 3** | `python simulations/fyp.py` | `Fyp.svg` |
| **Fig. 4** | `python simulations/finite_time.py` | `finite_time.svg` |
| **Fig. 5** | `python simulations/noise_comparison.py` | `amplitude_damping.svg` |
| **Fig. 6(a)** | `python threshold/sim.py` | `tolerable.svg` |
| **Fig. 6(b)** | `python threshold/grid.py` | `grid.svg` |
| **Fig. 7(a)** | `python applications/qkd_distance.py` | `qkd_distance.svg` |
| **Fig. 7(b)** | `python applications/repeater.py` | `repeater.svg` |

For other system sizes $N$ and sample budgets $K$, `variance_reduction_sampling.py` saves `fym_variance_reduced_nN_samplesK.svg` and `sampled_edp.py` saves `fym_nN_samplesK.svg`.
