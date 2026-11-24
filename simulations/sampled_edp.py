"""Sampled fidelity/yield versus measured pairs for four Hamiltonians.

Small CPU/GPU test: python simulations/sampled_edp.py
Figure 2 settings: python simulations/sampled_edp.py --n 10 --measured 2 3 4 5 6 --samples 3600
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from torch_edp import (
    build_ising, build_trapped_ion, build_rydberg, diagonal_eigenbasis,
    calcV, detect,
)
from utils import sample_pauli_iid
from recurrence import Fp, yd


def prepare_detection(U, m, device="cpu"):
    """Prepare the original detection kernel without enumerating all Paulis."""
    d = U.shape[0]
    n = d.bit_length() - 1
    if U.shape != (d, d) or d != 2**n or not 1 <= m < n:
        raise ValueError("Expected a 2**n square eigenbasis and 1 <= m < n")
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    basis = H
    for _ in range(n - 1):
        basis = np.kron(basis, H)
    UT = U.transpose()
    Udagger = UT.conjugate()
    v1, v2 = calcV(basis @ U.conjugate(), UT @ basis,
                  Udagger @ basis, basis @ U, m, n)
    return Udagger, torch.from_numpy(v1).to(device), torch.from_numpy(v2).to(device)


def sampled_result(U, m, p, samples, rng, device="cpu", *, progress=False, description="Pauli samples"):
    """Estimate fidelity/yield and first-order standard errors from iid Paulis."""
    if samples < 2 or not 0 <= p <= 1:
        raise ValueError("Require samples >= 2 and 0 <= p <= 1")
    n = U.shape[0].bit_length() - 1
    Udagger, v1, v2 = prepare_detection(U, m, device)
    rejection = np.empty(samples, dtype=float)
    for i in tqdm(range(samples), desc=description, unit="sample",
                  disable=not progress, position=1, leave=False, dynamic_ncols=True):
        _, matrix, _ = sample_pauli_iid(n, p, rng=rng)
        transformed = Udagger @ matrix.astype(np.complex64) @ U
        rejection[i] = float(np.real(detect(torch.from_numpy(transformed).to(device), v1, v2)))
    acceptance = 1 - rejection.mean()
    if acceptance <= 0:
        raise ValueError("Nonpositive estimated acceptance; increase the sample count")
    acceptance_se = np.std(rejection, ddof=1) / np.sqrt(samples)
    fidelity = ((1 - 3*p/4)**n / acceptance)**(1/(n-m))
    return {
        "acceptance": float(acceptance),
        "acceptance_se": float(acceptance_se),
        "fidelity": float(fidelity),
        "fidelity_se": float(fidelity * acceptance_se / ((n-m) * acceptance)),
        "yield": float(acceptance * (n-m)/n),
        "yield_se": float(acceptance_se * (n-m)/n),
    }


def build_eigenbases(n):
    """Paper Hamiltonian parameters; periodic Ising and Rydberg spacing 6."""
    return {
        "Ising": np.linalg.eigh(build_ising(N=n, J0=1, B=2))[1],
        "Diagonal": diagonal_eigenbasis(N=n),
        "Rydberg": np.linalg.eigh(build_rydberg(N=n, Omega=1.0, detuning=2.5,
                                               C=862890, r0=6.0))[1],
        "Trapped-ion": np.linalg.eigh(build_trapped_ion(N=n, J0=1, B=2, alpha=1))[1],
    }


def plot_results(rows, output, p):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    right = ax.twinx()
    colors = {"Ising": "C0", "Diagonal": "C4", "Rydberg": "C1", "Trapped-ion": "C6"}
    for method, color in colors.items():
        data = [row for row in rows if row["method"] == method]
        ms = [row["m"] for row in data]
        ax.errorbar(ms, [r["fidelity"] for r in data],
                    yerr=[r["fidelity_se"] for r in data], fmt="o-", capsize=3,
                    color=color, label=method)
        right.errorbar(ms, [r["yield"] for r in data],
                       yerr=[r["yield_se"] for r in data], fmt="x--", capsize=3, color=color)
    F0 = 1 - 3*p/4
    ax.axhline(Fp(F0), color="C3", label="Recurrence")
    right.axhline(yd(F0), color="C3", linestyle="--")
    ax.set_xlabel("measured qubits m")
    ax.set_ylabel("normalized fidelity")
    right.set_ylabel("yield")
    ax.set_xticks(sorted({row["m"] for row in rows}))
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.22), ncol=3, frameon=False)
    right.legend(handles=[Line2D([], [], color="black", marker="o", label="Fidelity"),
                          Line2D([], [], color="black", marker="x", linestyle="--", label="Yield")],
                 loc="upper right", bbox_to_anchor=(1, 1.22), frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--measured", type=int, nargs="+", default=[2, 3,4])
    parser.add_argument("--p", type=float, default=0.2)
    parser.add_argument("--samples", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=23333)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.n < 2 or any(m < 1 or m >= args.n for m in args.measured):
        parser.error("Require n >= 2 and 1 <= each measured count < n")
    if args.samples < 2 or not 0 <= args.p <= 1:
        parser.error("Require samples >= 2 and 0 <= p <= 1")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    out = Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"fym_n{args.n}_samples{args.samples}"
    rows = []
    print(f"Building and diagonalizing Hamiltonians for n={args.n}...", flush=True)
    eigenbases = build_eigenbases(args.n)
    # Overall ETA includes detection preparation and checkpointing, once jobs finish.
    with tqdm(total=len(eigenbases) * len(set(args.measured)), desc="Sweep",
              unit="calculation", position=0, dynamic_ncols=True) as sweep:
        for index, (method, U) in enumerate(eigenbases.items()):
            for m in sorted(set(args.measured)):
                # Separate deterministic stream per method and m; sweep order does not affect it.
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, index, m]))
                sweep.set_description(f"Sweep: {method}, m={m}")
                result = sampled_result(U, m, args.p, args.samples, rng, device,
                                        progress=True, description=f"{method}, m={m}")
                rows.append(dict(method=method, n=args.n, m=m, p=args.p,
                                 samples=args.samples, seed=args.seed, device=device, **result))
                sweep.update(1)
    plot_results(rows, out / f"{stem}.svg", args.p)
    print(f"Saved {out / (stem + '.svg')}")


if __name__ == "__main__":
    main()
