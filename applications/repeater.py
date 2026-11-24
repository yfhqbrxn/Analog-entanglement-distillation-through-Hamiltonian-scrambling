"""Figure 7(b): repeater fidelity versus distance, plus shared phase-error helpers."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from qkd_utils import calcETA, calcE1, Y0, e0, ed

beta = 2.82 * 10**-3
def calcEp(l):
    # eta = calcETA(l)
    # e1 = (eta*ed + (1-eta)*Y0*e0) / (eta + (1-eta) * Y0)
    # return e1 
    eta = calcETA(l)
    ept = (1 - np.exp(-beta*l/2)) / 2
    ep = (eta*ept + (1-eta)*Y0*e0) / (eta + (1-eta) * Y0)
    return ep

def p_recurrence(q00, q01, q10, q11, r):
    yd = 1
    ls = sorted([q01, q10, q11])
    p00 = q00
    p01 = ls[1]
    p10 = ls[0]
    p11 = ls[2]

    for _ in range(r):
        p_pass = (p00 + p10)**2 + (p01+p11)**2
        r00 = (p00**2 + p10**2) / p_pass 
        r01 = 2 * p00 * p10 / p_pass 
        r10 = (p01**2 + p11**2) / p_pass 
        r11 = 2 * p01 * p11 / p_pass

        # print(q00, q01, q10, q11, p_pass)

        ls = sorted([r01, r10, r11])
        # print(ls)

        p00 = r00
        p01 = ls[2]
        p10 = ls[0]
        p11 = ls[1]

        yd = yd * p_pass / 2

    # print(p00, p01, p10, p11)

    return p00, p01, p10, p11


def Frec(l, iter=1):
    eb = calcE1(l, ed)
    ep = calcEp(l)
    p00, p01, p10, p11 = p_recurrence((1-eb)*(1-ep), eb*(1-ep), ep*(1-eb), ep*eb, iter)
    return p00

def Fham(l, n=15, m=12):
    eb = calcE1(l, ed)
    ep = calcEp(l)
    
    numerator = ((1-eb)*(1-ep))**n
    # denom = (2**-m) * (1 - (1-ep)**n) + (1-ep)**n * (1-eb)**m
    denom = numerator + (1 - numerator) * (2**-m)

    F = numerator / denom

    return F

if __name__ == "__main__":
    # Preserve the original figure style without changing styles on import.
    try:
        import seaborn as sns
        sns.set_theme(style="darkgrid")
    except ImportError:
        plt.style.use("ggplot")

    ls = np.linspace(100, 330, 100)
    Fones = []
    Ftwos = []
    Fhams = []

    for l in ls:
        Fones.append(Frec(l, iter=1))
        Ftwos.append(Frec(l, iter=2))
        Fhams.append(Fham(l))

    plt.figure()
    plt.plot(ls, Fones, label='1-round recurrence')
    plt.plot(ls, Ftwos, label='2-round recurrence')
    plt.plot(ls, Fhams, color='orange', label='Hamiltonian')
    plt.plot(ls, (1-calcE1(ls, ed)) * (1-calcEp(ls)), label='no distillation')
    plt.axhline(y=0.913, color='gray', linestyle='--', alpha=0.4)
    plt.legend(fontsize=12)

    plt.xlabel('distance (km)', fontsize=14)
    plt.ylabel('fidelity', fontsize=14)
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "repeater.svg", bbox_inches="tight", pad_inches=0.12)

    # plt.show()
