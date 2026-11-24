"""Generate the fidelity/yield versus depolarizing-strength plot.

Run from the AEDP root: python simulations/fyp.py
Output: outputs/Fyp.svg
The current experiment uses N=6, m=3, and
10 noise strengths from 0.05 to 0.3. Uses CUDA automatically when available (falls back to CPU).
"""

from pathlib import Path


def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from recurrence import Fp, yd
    from torch_edp import (
        build_ising, build_trapped_ion, build_rydberg, diagonal_eigenbasis,
        prepare_edp, process_unitary,
    )

    # Experiment settings; N=6 overrides the builders' N=5 default.
    N = 6
    J0 = 1
    B = 2
    alpha = 1
    Omega = 1.0
    detuning = 2.5
    C = 862890
    r0 = 6.0
    np.random.seed(23333)

    Ising = build_ising(N=N, J0=J0, B=B)
    TP = build_trapped_ion(N=N, J0=J0, B=B, alpha=alpha)
    Rydberg = build_rydberg(N=N, Omega=Omega, detuning=detuning, C=C, r0=r0)
    _, Uis = np.linalg.eigh(Ising)
    _, Uryd = np.linalg.eigh(Rydberg)
    _, Utp = np.linalg.eigh(TP)
    Uid = diagonal_eigenbasis(N=N)
    context = prepare_edp(N=N)

    nss = np.linspace(0.05, 0.3, 10)
    ns = np.array([0.2])
    ms = [1,2,3,4]
    m = 3
    fs_is = []
    fs_id = []
    fs_ryd = []
    prs_is = []
    prs_id = []
    prs_ryd = []

    fs_rec2 = []
    prs_rec2 = []

    fs_rec1 = []
    prs_rec1 = []

    # for m in ms:
    for ns in nss:
        ns = np.array([ns])
        pr_is, f_is = process_unitary(Uis, m, ns, context=context)
        fs_is.append(f_is)
        prs_is.append(pr_is)
        pr_id, f_id = process_unitary(Uid, m, ns, context=context)
        fs_id.append(f_id)
        prs_id.append(pr_id)
        pr_ryd, f_ryd = process_unitary(Uryd, m, ns, context=context)
        fs_ryd.append(f_ryd)
        prs_ryd.append(pr_ryd)

        pr_rec1, f_rec1 = process_unitary(Utp, m, ns, context=context)
        fs_rec1.append(f_rec1)
        prs_rec1.append(pr_rec1)

        F0 = Fp(1 - 3*ns.item()/4)
        num_iter = 2
        yr = yd(1 - 3*ns.item()/4)

        # fs_rec1.append(F0)
        # prs_rec1.append(yr)

        # for _ in range(num_iter-1):
        #   yr = yr * yd(F0)
        #   F0 = Fp(F0)  

        # fs_rec2.append(F0)
        # prs_rec2.append(yr)
        
        # det_pr_id, det_rate_id, f_id = process_unitary(Uid, m, ns, context=context)
        # det_pr_is, det_rate_is, f_is = process_unitary(Uis, m, ns, context=context)

    print(nss)

    # print(fs_is, fs_id, fs_ryd)
    # print(fs_rec2, prs_rec2)


    # flatten prob arrays (ns has length 1 here)
    prs_id_vals = np.array([np.real(p[0]) if np.ndim(p) else np.real(p) for p in prs_id])
    prs_is_vals = np.array([np.real(p[0]) if np.ndim(p) else np.real(p) for p in prs_is])
    prs_ryd_vals = np.array([np.real(p[0]) if np.ndim(p) else np.real(p) for p in prs_ryd])
    prs_rec1 = np.array([np.real(p[0]) if np.ndim(p) else np.real(p) for p in prs_rec1])

    # print(prs_is_vals, prs_id_vals, prs_ryd_vals)
    print(prs_rec1)

    color_id = 'C4'
    color_is = 'C0'
    color_ryd = 'C1'
    color_rec = 'C3'
    color_rec1 = 'C6'

    fig, ax_left = plt.subplots(figsize=(6, 4))

    # xis = ms
    xis = nss

    fid_id_sc = ax_left.scatter(xis, fs_id, label='Diagonal fidelity', color=color_id, marker='o')
    fid_is_sc = ax_left.scatter(xis, fs_is, label='Ising fidelity', color=color_is, marker='o')
    fid_ryd_sc = ax_left.scatter(xis, fs_ryd, label='Rydberg fidelity', color=color_ryd, marker='o')
    # fid_rec_sc = ax_left.scatter(xis, fs_rec2, label='Recurrence fidelity', color=color_rec, marker='o')
    fid_rec1_sc = ax_left.scatter(xis, fs_rec1, label='Trapped-ion fidelity', color=color_rec1, marker='o')
    fid_id_ln, = ax_left.plot(xis, fs_id, linestyle='-', linewidth=1.0, color=color_id)
    fid_is_ln, = ax_left.plot(xis, fs_is, linestyle='-', linewidth=1.0, color=color_is)
    # fid_rec_ln, = ax_left.plot(xis, fs_rec2, linestyle='-', linewidth=1.0, color=color_rec)
    fid_rec1_ln, = ax_left.plot(xis, fs_rec1, linestyle='-', linewidth=1.0, color=color_rec1)
    fid_ryd_ln, = ax_left.plot(xis, fs_ryd, linestyle='-', linewidth=1.0, color=color_ryd)
    baseline = 1 - nss * 3 / 4
    # baseline = 1 - ns.item() / 2
    baseline_ln, = ax_left.plot(xis, baseline, linestyle='-', marker='o', color='gray', linewidth=0.8, label='No distillation')
    # baseline_ln = ax_left.axhline(y=baseline, linestyle='--', color='gray', linewidth=0.8, label='No distillation')
    # ax_left.annotate('initial fidelity', xy=(ms[0], baseline), xytext=(ms[0]+0.1, baseline-0.01),
                    #  textcoords='data', fontsize=9, color='gray')
    # ax_left.set_xlabel('measured qubits m')
    ax_left.set_xlabel('depolarizing strength $p$')
    # ax_left.set_xticks([1, 2, 3, 4])
    ax_left.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ax_left.set_ylabel('fidelity')
    ax_left.grid(alpha=0.3, linestyle='--', linewidth=0.6)

    ax_right = ax_left.twinx()
    pr_id_sc = ax_right.scatter(xis, prs_id_vals, color=color_id, marker='x', label='Diagonal yield', zorder=3)
    pr_is_sc = ax_right.scatter(xis, prs_is_vals, color=color_is, marker='x', label='Ising yield', zorder=3)
    pr_ryd_sc = ax_right.scatter(xis, prs_ryd_vals, color=color_ryd, marker='x', label='Rydberg yield', zorder=3)
    # pr_rec_sc = ax_right.scatter(xis, prs_rec2, color=color_rec, marker='x', label='Recurrence yield', zorder=3)
    pr_rec1_sc = ax_right.scatter(xis, prs_rec1, color=color_rec1, marker='x', label='Trapped-ion yield', zorder=3)
    pr_id_ln, = ax_right.plot(xis, prs_id_vals, linestyle='--', linewidth=1.0, color=color_id)
    pr_is_ln, = ax_right.plot(xis, prs_is_vals, linestyle='--', linewidth=1.0, color=color_is)
    pr_ryd_ln, = ax_right.plot(xis, prs_ryd_vals, linestyle='--', linewidth=1.0, color=color_ryd)
    # pr_rec_ln, = ax_right.plot(xis, prs_rec2, linestyle='--', linewidth=1.0, color=color_rec)
    pr_rec1_ln, = ax_right.plot(xis, prs_rec1, linestyle='--', linewidth=1.0, color=color_rec1)
    ax_right.set_ylabel('yield')
    ax_right.set_ylim(ax_right.get_ylim()[0], 0.6)

    # split legend: colors -> Hamiltonian, markers -> metric
    color_handles = [
        Line2D([0], [0], color=color_id, lw=2, label='Diagonal'),
        Line2D([0], [0], color=color_is, lw=2, label='Ising'),
        Line2D([0], [0], color=color_ryd, lw=2, label='Rydberg'),
        Line2D([0], [0], color=color_rec1, lw=2, label='Trapped-ion'),
        # Line2D([0], [0], color=color_rec, lw=2, label='2-Recurrence'),
        Line2D([0], [0], color='gray', lw=2, label='Initial', linestyle='-')
        # baseline_ln,
    ]
    marker_handles = [
        Line2D([0], [0], marker='o', linestyle='-', color='gray', markersize=6, label='Fidelity'),
        Line2D([0], [0], marker='x', linestyle='--', color='gray', markersize=6, label='Yield'),
    ]
    color_legend = ax_left.legend(color_handles, [h.get_label() for h in color_handles],
                                  loc='upper left', bbox_to_anchor=(-0.08, 1.21),
                                  frameon=False, facecolor='white', ncol=3, columnspacing=1.0, handlelength=2.2)
    ax_left.add_artist(color_legend)
    marker_legend = ax_left.legend(marker_handles, [h.get_label() for h in marker_handles],
                                   loc='upper right', bbox_to_anchor=(1.05, 1.21),
                                   frameon=False, facecolor='white', ncol=1, columnspacing=1.2, handlelength=1.5)
    fig.subplots_adjust(top=0.82, bottom=0.16)

    # ax_left.set_title('n=5, noise_strength=0.05, original fidelity is 0.9625')
    output = Path(__file__).resolve().parents[1] / "outputs" / "Fyp.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.12)
    print(f"Saved to {output}")
    # plt.show()


if __name__ == "__main__":
    main()
