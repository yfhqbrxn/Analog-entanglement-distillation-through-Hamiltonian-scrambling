"""Figure 7(a): maximum QKD distance versus photon bit error rate."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from qkd_utils import H, calcE1, f, find_zero_distance
from repeater import calcEp, p_recurrence

def Rate_rec(l, ed, iter=1):
    eb = calcE1(l, ed)
    ep = calcEp(l)
    p00, p01, p10, p11 = p_recurrence((1-eb)*(1-ep), eb*(1-ep), ep*(1-eb), ep*eb, iter)
    ebp = p01 + p11
    epp = p10 + p11
    if ebp > epp:
        ebp, epp = epp, ebp
    return 1 - f*H(ebp) - H(epp)


def Rate_Hamiltonian(l, ed, n=15, m=12):
    ep = calcE1(l, ed)         # in Hadamard basis, 
    eb = calcEp(l)             # shift X and Z.
    
    # eb = eb / (1 + (1-p/2)**(n-m) * ((2-3*p/2)**m - (1-p/2)**m))
    ebp = eb / (1 - (1-eb)**(n-m) + 2**m * (1-eb)**n * (1-ep)**m)

    numerator = (2**-m) * (1 - (1-eb)**(n-m)) / 2 + (1-eb)**n * (1-ep)**m * ep
    denom = (2**-m) * (1 - (1-eb)**(n-m)) + (1-eb)**n * (1-ep)**m

    epp = numerator / denom

    return 1 - f*H(ebp) - H(epp)

def Rate_one(l, ed):
    eb = calcE1(l, ed)
    ep = calcEp(l)
    return 1 - f*H(eb) - H(ep)

if __name__ == "__main__":
    # Preserve the original figure style without changing styles on import.
    try:
        import seaborn as sns
        sns.set_theme(style="darkgrid")
    except ImportError:
        plt.style.use("ggplot")

    eds = np.linspace(0.01, 0.07, 100)
    Rone = np.array([find_zero_distance(Rate_one, ed) for ed in eds])
    Rtwo = np.array([find_zero_distance(Rate_rec, ed) for ed in eds])
    Rate_sec = partial(Rate_rec, iter=2)
    Rsec = np.array([find_zero_distance(Rate_sec, ed) for ed in eds])
    Rham = np.array([find_zero_distance(Rate_Hamiltonian, ed) for ed in eds])

    fig, ax = plt.subplots()
    ax.plot(eds, Rone, label="one-way hashing",  color="#00bcd4")
    ax.plot(eds, Rtwo, label="1-round recurrence",  color="#e91e63", linestyle="-")
    ax.plot(eds, Rsec, label="2-round recurrence",  color="purple", linestyle="-")
    ax.plot(eds, Rham, label="Hamiltonian", color="#ff9800", linestyle="-")

    ax.set_xlabel("Photon bit error rate $e_d$", fontsize=14)
    ax.set_ylabel("Maximum transmission distance (km)", fontsize=14)
    # ax.set_title("Maximum Secure Distance vs Bit Error ", fontsize=16, fontweight="bold")
    ax.grid(which="both", linestyle=":", linewidth=0.8, alpha=0.6)
    # ax.set_xlim(eds.min(), eds.max())
    ax.set_ylim(240, 420)
    # ax.yaxis.set_major_formatter(plt.LogFormatterMathtext())
    # ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.legend(frameon=True, loc="upper right", fontsize=12)

    fig.tight_layout()
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "qkd_distance.svg", bbox_inches="tight", pad_inches=0.12)
    # plt.show()