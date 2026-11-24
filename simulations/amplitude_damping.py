import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from qiskit.quantum_info import Pauli
from numba import njit
from tqdm import tqdm
import torch
import sys

np.random.seed(23333)
eps = 1e-5

def rand_unitary(n):
    d = 2**n
    z = np.random.randn(d, d) + 1j*np.random.randn(d, d)
    q, r = np.linalg.qr(z)
    return q * (r.diagonal() / np.abs(r.diagonal()))

def int2vec(x, n):
    bits = f"{x:0{n}b}"
    return np.array(list(bits), dtype=np.complex128)

def paulis(n):
    labels = [''.join(p) for p in product('IXYZ', repeat=n)]
    mats   = [Pauli(s).to_matrix() for s in labels]
    return labels, mats

N = 5
d = 2**N

H  = np.array([[1, 1],
               [1,-1]], dtype=complex)/np.sqrt(2)
H5H = H   # also 32x32 matrix
H5I = np.eye(2)
for i in range(N-1):
    H5H = np.kron(H5H, H)
    H5I = np.kron(H5I, np.eye(2))

# U = rand_unitary(N)

Uid = np.zeros((d, d), dtype=np.complex128)
for i in range(d):
    Uid[i, i] = 1.0


# TFIM
Ising = np.zeros((d, d), dtype=np.complex128)
h = 0.5
# J = [0.2, 0.35, 0.51, 0.19, 0.33, 0.52, 0.97, 0.43, 0.16]
J = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
sJ = 0.2

for i in range(N):
    j = (i+1) % N
    Ising += Pauli(''.join('X' if k in (i, j) else 'I' for k in range(N))).to_matrix() * h
    Ising += Pauli(''.join('Z' if k in (i, ) else 'I' for k in range(N))).to_matrix() * J[i] 

# superconducting
superconducting = np.zeros((d, d), dtype=np.complex128)

for i in range(N):
    j = (i+1) % N
    superconducting += Pauli(''.join('X' if k in (i, j) else 'I' for k in range(N))).to_matrix() * h
    superconducting += Pauli(''.join('Y' if k in (i, j) else 'I' for k in range(N))).to_matrix() * h
    superconducting += Pauli(''.join('Z' if k in (i, ) else 'I' for k in range(N))).to_matrix() * sJ

# # Rydberg
# Rydberg = np.zeros((d, d), dtype=np.complex128)

# Omega = 0.1
# cf = []
# for i in range(N):
#     cf.append()

# for i in range(N):
#     j = (i+1) % N
#     Rydberg += Pauli(''.join('Z' if k in (i, ) else 'I' for k in range(N))).to_matrix() * cf[i]
#     Rydberg += Pauli(''.join('X' if k in (i, ) else 'I' for k in range(N))).to_matrix() * Omega


# H should be Hermitian: H == H.conj().T
w, Uis = np.linalg.eigh(Ising)   # w: eigenvalues, U: eigenvectors (columns)
_, Usc = np.linalg.eigh(superconducting)


@njit(nopython=True)
def calcV(XU, UTX, UdaggerX, XUstar, m):
    v1 = np.zeros((2**m, d), dtype=np.complex128)
    v2 = np.zeros((d, d), dtype=np.complex128)

    print("Begin preparation")

    for b in range(2**m):
        for y in range(2**(N-m)):
            for i in range(d):
                by = b*(2**(N-m)) + y 
                v1[b, i] += UTX[i, by] * UdaggerX[i, by]
        
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            for b in range(2**m):
                for x in range(2**(N-m)):
                    for y in range(2**(N-m)):
                        bx = b*(2**(N-m)) + x
                        by = b*(2**(N-m)) + y 
                        v2[i, j] += XUstar[bx, i] * XU[bx, j] * UTX[i, by] * UdaggerX[j, by]
    
    print("Preparation done")
    return v1, v2

def detect(tp, v1, v2):
    O = tp * tp.t()
    dg = torch.diag(tp)
    Q = torch.outer(dg, dg)

    res = (v2 * Q).sum()
    # for b in range(2**m):
    #     res += v1[b, :].conj().t() @ O @ v1[b, :]
    ##################################################
    tmp = v1 @ O @ v1.conj().t()   # shape (B, B)   
    res += torch.trace(tmp)         # sum over b
    ##################################################

    res /= d
    return 1 - res.item() 

labels, mats  = paulis(N)
weights = [sum(c != 'I' for c in s) for s in labels]
weightsI = [sum(c == 'I' for c in s) for s in labels]
weightsX = [sum(c == 'X' for c in s) for s in labels]
weightsY = [sum(c == 'Y' for c in s) for s in labels]
weightsZ = [sum(c == 'Z' for c in s) for s in labels]

def amplitude_damping_weight(p, gamma, wtI, wtx, wty, wtz):
    pI = (1 + 2 * (1-p) *np.sqrt(1-gamma) + (1-p)*(1-gamma)) / 4
    px = (1 - (1-p)*(1-gamma)) / 4
    py = (1 - (1-p)*(1-gamma)) / 4
    pz = (1 - 2 * (1-p) *np.sqrt(1-gamma) + (1-p)*(1-gamma)) / 4
    ret = (pI**wtI) * (px**wtx) * (py**wty) * (pz**wtz)
    return ret

def process_unitary(U, m, ns, gamma=0.02, hadamard_basis=False):
    UT = U.transpose()
    Udagger = UT.conjugate()
    Ustar = U.conjugate()

    if hadamard_basis:
        H5 = H5H
    else:
        H5 = H5I

    XUstar = H5 @ Ustar
    XU = H5 @ U
    UTX = UT @ H5
    UdaggerX = Udagger @ H5

    det_pr = np.zeros_like(ns, dtype=np.complex128)

    import time
    start = time.perf_counter()

    v1, v2 = calcV(XU, UTX, UdaggerX, XUstar, m)

    end = time.perf_counter()

    elapsed = end - start
    print("Elapsed:", elapsed, "seconds")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    v1_t = torch.from_numpy(v1).to(device)
    v2_t = torch.from_numpy(v2).to(device)
    for i in tqdm(range(4**N)):
        wt = weights[i]
        wtI = weightsI[i]
        wtx = weightsX[i]
        wty = weightsY[i]
        wtz = weightsZ[i]

        P = mats[i]
        tp = Udagger @ P @ U
        ret = detect(torch.from_numpy(tp).to(device), v1_t, v2_t)
        for i, noise in enumerate(ns):
            prob = amplitude_damping_weight(noise, gamma, wtI, wtx, wty, wtz)
            det_pr[i] += prob * ret
            
    # det_rate = np.real(det_pr / ((1-(1-(3*ns/4))**N)))
    p_intact = ((1 + 2 * (1-ns) *np.sqrt(1-gamma) + (1-ns) * (1-gamma)) / 4)**N
    print(p_intact)
    det_rate = np.real(det_pr / (1 - p_intact))
    fidelity = (p_intact / (1 - det_pr))**(1/(N-m))

    # return det_pr, det_rate, fidelity
    return np.real(fidelity.item()), 1-det_pr.item()

if __name__ == "__main__":

    ns = np.array([0.2])
    ms = [1,2,3,4]
    fs_is = []
    fs_id = []
    fs_sc = []
    for m in ms:
        fs_is.append(process_unitary(Uis, m, ns))
        fs_id.append(process_unitary(Uid, m, ns))
        fs_sc.append(process_unitary(Usc, m, ns))

    print(fs_is, fs_id)
    gamma = 0.05
    plt.figure()
    plt.scatter(ms, fs_id, label='Diagonal Hamiltonian')
    plt.scatter(ms, fs_is, label='Ising Hamiltonian', color='orange')
    plt.scatter(ms, fs_sc, label='superconducting', color='purple')
    plt.plot(ms, fs_id, linestyle='--', linewidth=0.5)
    plt.plot(ms, fs_is, color='orange', linestyle='--', linewidth=0.5)
    plt.plot(ms, fs_sc, linestyle='--', linewidth=0.5, color='purple')
    plt.legend()
    plt.xlabel('measured qubits m')
    plt.xticks([1, 2, 3, 4])  # positions and optional labels
    f_origin = (1 + 2 * (1-ns) *np.sqrt(1-gamma) + (1-ns)*(1-gamma)) / 4
    plt.axhline(y=f_origin, linestyle='--')   # dashed line
    plt.ylabel('fidelity')
    plt.title('n=5, noise_strength=0.05, original fidelity is about 0.953')
    plt.show()
    # plt.savefig('fidelity_m.png')

    # CODE FOR GENERATING FIDELITY_NOISE FIGURE

    # plt.figure()
    # plt.plot(ns, f_id, label='Diagonal Hamiltonian')
    # plt.plot(ns, f_is, label='Ising Hamiltonian', color='orange')
    # plt.plot(ns, 1-3*ns/4, label='before distillation')
    # plt.ylabel("average fidelity")
    # plt.xlabel("local depolarizing noise strength")
    # plt.title("n={},m={}".format(N, m))
    # plt.legend()
    # # plt.show()
    # plt.savefig('fidelity_noise.png')



