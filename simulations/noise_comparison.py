from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Pauli
from amplitude_damping import process_unitary
from infinite_time import main

N = 5
m = 3
d = 2**N

p = 0.2
use_hadamard= True

gammas = np.linspace(0, 0.2, 10).tolist()
fds_Is_wo, yields_Is_wo, fds_Ryd_wo, yields_Ryd_wo, fds_initial = main(N, m, gammas, p, hadamard=use_hadamard)

# TFIM
Ising = np.zeros((d, d), dtype=np.complex128)

J0 = 1
B = 2

for i in range(N):
    j = (i+1) % N
    Ising += Pauli(''.join('X' if k in (i, j) else 'I' for k in range(N))).to_matrix() * J0
    Ising += Pauli(''.join('Z' if k in (i, ) else 'I' for k in range(N))).to_matrix() * B 


TP = np.zeros((d, d), dtype=np.complex64)
alpha = 1

for i in range(N):
    for j in range(i+1, N):
        TP += Pauli(''.join('X' if k in (i, j) else 'I' for k in range(N))).to_matrix() * J0 / (abs(j-i)**alpha)
    TP += Pauli(''.join('Z' if k in (i, ) else 'I' for k in range(N))).to_matrix() * B 

# Rydberg
Rydberg = np.zeros((d, d), dtype=np.complex128)

Omega = 1
detuning = 2.5
C = 862890
r0 = 6.0

def Vij(i, j):
    return C / ((abs(i - j) * r0)**6)

Vi = [0 for _ in range(N)]
for i in range(N):
    for j in range(N):
        if i != j:
            Vi[i] += Vij(i, j)

print(Vi)

for i in range(N):
    Rydberg += Pauli(''.join('X' if k in (i, ) else 'I' for k in range(N))).to_matrix() * Omega
    Rydberg += Pauli(''.join('Z' if k in (i, ) else 'I' for k in range(N))).to_matrix() * (-detuning/2+Vi[i]/4)

for i in range(N):
    for j in range(N):
        if i == j:
            continue
        coeff = Vij(i, j) / 4
        Rydberg += Pauli(''.join('Z' if k in (i, j) else 'I' for k in range(N))).to_matrix() * coeff / 4


# H should be Hermitian: H == H.conj().T
w, Uis = np.linalg.eigh(Ising)   # w: eigenvalues, U: eigenvectors (columns)
_, Uryd = np.linalg.eigh(Rydberg)
_, Utp = np.linalg.eigh(TP)

ns = np.array([p])

fds_Is_w = []
yields_Is_w = []
fds_Ryd_w = []
yields_Ryd_w = []

for gamma in gammas:
    # fd_Is, yield_Is = process_unitary(Uis, m, ns, gamma, hadamard_basis=use_hadamard)
    fd_Is, yield_Is = process_unitary(Utp, m, ns, gamma, hadamard_basis=use_hadamard)
    fd_Ryd, yield_Ryd = process_unitary(Uryd, m, ns, gamma, hadamard_basis=use_hadamard)
    fds_Is_w.append(fd_Is)
    fds_Ryd_w.append(fd_Ryd)
    yields_Is_w.append(yield_Is)
    yields_Ryd_w.append(yield_Ryd)

print(fds_Is_w, fds_Ryd_w, fds_Is_wo, fds_Ryd_wo, fds_initial)
fig, ax = plt.subplots(figsize=(7, 4))

fid_is_tw, = ax.plot(gammas, fds_Is_w, color='C0', marker='o', label='Trapped-ion (with P.T)')
fid_ryd_tw, = ax.plot(gammas, fds_Ryd_w, color='C1', marker='s', label='Rydberg (with P.T)')
fid_is_no, = ax.plot(gammas, fds_Is_wo, color='C0', linestyle='--', marker='^', label='Trapped-ion (no P.T)')
fid_ryd_no, = ax.plot(gammas, fds_Ryd_wo, color='C1', linestyle='--', marker='x', label='Rydberg (no P.T)')

baseline = np.array(fds_initial)
baseline_ln, = ax.plot(gammas, baseline, color='gray', linestyle=':', linewidth=1.2, label='Before distillation')

ax.set_xlabel(r'Amplitude damping noise strength $\gamma$')
ax.set_ylabel('fidelity')
ax.set_ylim(0.75, 0.95)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.6)

handles = [fid_is_tw, fid_ryd_tw, fid_is_no, fid_ryd_no, baseline_ln]
ax.legend(handles, [h.get_label() for h in handles], loc='lower left',
          frameon=True, facecolor='white')

fig.tight_layout()
output_dir = Path(__file__).resolve().parents[1] / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "amplitude_damping.svg", bbox_inches="tight", pad_inches=0.12)
# plt.show()
