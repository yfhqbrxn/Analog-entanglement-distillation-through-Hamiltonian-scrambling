"""Shared exact-Pauli EDP calculation for the configured Hamiltonians.

Run fyp.py for the fidelity/yield versus noise-strength figure.
Hamiltonians and Pauli matrices are prepared only by explicit function calls.
The builder defaults preserve the original experiment parameters.
"""

import numpy as np
from itertools import product
from dataclasses import dataclass
from qiskit.quantum_info import Pauli
from numba import njit, prange
from tqdm import tqdm
import torch
pi = np.pi

eps = 1e-5

def rand_unitary(n):
    d = 2**n
    z = np.random.randn(d, d) + 1j*np.random.randn(d, d)
    q, r = np.linalg.qr(z)
    return q * (r.diagonal() / np.abs(r.diagonal()))

def int2vec(x, n):
    bits = f"{x:0{n}b}"
    return np.array(list(bits), dtype=np.complex64)

def paulis(n):
    labels = [''.join(p) for p in product('IXYZ', repeat=n)]
    mats   = [Pauli(s).to_matrix() for s in labels]
    return labels, mats

def build_ising(N=5, J0=1, B=2):
    """Periodic transverse-field Ising Hamiltonian."""
    d = 2**N
    Ising = np.zeros((d, d), dtype=np.complex64)


    for i in range(N):
        j = (i+1) % N
        Ising += Pauli(''.join('X' if k in (i, j) else 'I' for k in range(N))).to_matrix() * J0
        Ising += Pauli(''.join('Z' if k in (i, ) else 'I' for k in range(N))).to_matrix() * B
    return Ising

def build_trapped_ion(N=5, J0=1, B=2, alpha=1):
    """Power-law trapped-ion Hamiltonian."""
    d = 2**N
    TP = np.zeros((d, d), dtype=np.complex64)

    for i in range(N):
        for j in range(i+1, N):
            TP += Pauli(''.join('X' if k in (i, j) else 'I' for k in range(N))).to_matrix() * J0 / (abs(j-i)**alpha)
        TP += Pauli(''.join('Z' if k in (i, ) else 'I' for k in range(N))).to_matrix() * B
    return TP

def build_rydberg(N=5, Omega=1.0, detuning=2.5, C=862890, r0=6.0):
    """Rydberg Hamiltonian; preserves the original interaction convention."""
    d = 2**N
    Rydberg = np.zeros((d, d), dtype=np.complex64)


    def Vij(i, j):
        return C / ((abs(i - j) * r0)**6)

    Vi = [0 for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                Vi[i] += Vij(i, j)

    for i in range(N):
        Rydberg += Pauli(''.join('X' if k in (i, ) else 'I' for k in range(N))).to_matrix() * Omega
        Rydberg += Pauli(''.join('Z' if k in (i, ) else 'I' for k in range(N))).to_matrix() * (-detuning/2+Vi[i]/4)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            coeff = Vij(i, j) / 4
            Rydberg += Pauli(''.join('Z' if k in (i, j) else 'I' for k in range(N))).to_matrix() * coeff / 4
    return Rydberg

def diagonal_eigenbasis(N=5):
    """Computational eigenbasis for the diagonal-Hamiltonian comparison."""
    return np.eye(2**N, dtype=np.complex64)


@dataclass
class EDPContext:
    N: int
    H5: np.ndarray
    mats: list
    weights: list


def prepare_edp(N=5):
    """Prepare the Hadamard basis and Pauli matrices once per experiment."""
    if not isinstance(N, int) or N < 1:
        raise ValueError("N must be a positive integer")
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    H5 = H
    for i in range(N - 1):
        H5 = np.kron(H5, H)
    labels, mats = paulis(N)
    weights = [sum(c != 'I' for c in label) for label in labels]
    return EDPContext(N, H5, mats, weights)


@njit(fastmath=True, parallel=True)
def optimized_numba_sum(d, m, N, XUstar, XU, UTX, UdaggerX, v2):
    """
    Optimized summation using Numba with mathematical factorization.
    Complexity reduced from O(d^2 * 2^N * 2^(N-m)) to O(d^2 * 2^N).
    """
    
    # Pre-calculate constants
    B = 2**m           # Number of blocks
    K = 2**(N - m)     # Block size (range of x and y)
    
    # Parallelize over the output rows (i)
    for i in prange(d):
        for j in range(d):
            
            # Skip diagonal as per original code
            if i == j:
                continue
            
            block_sum = 0.0 + 0.0j  # Accumulator for the sum over blocks (b)
            
            # Loop over blocks
            for b in range(B):
                
                # --- Factorization Magic Happens Here ---
                # Instead of nesting x inside y, we compute them side-by-side.
                # sum(A * B) = sum(A) * sum(B)
                
                # 1. Calculate Sum over x
                sum_x = 0.0 + 0.0j
                base_idx = b * K
                for x in range(K):
                    bx = base_idx + x
                    sum_x += XUstar[bx, i] * XU[bx, j]

                # 2. Calculate Sum over y
                sum_y = 0.0 + 0.0j
                for y in range(K):
                    by = base_idx + y
                    sum_y += UTX[i, by] * UdaggerX[j, by]

                # 3. Multiply the independent sums
                block_sum += sum_x * sum_y
            
            # Update the result matrix
            v2[i, j] += block_sum
            
    return v2

@njit(nopython=True)
def calcV(XU, UTX, UdaggerX, XUstar, m, N):
    d = XU.shape[0]
    # memory still d**2, affordable
    v1 = np.zeros((2**m, d), dtype=np.complex64)
    v2 = np.zeros((d, d), dtype=np.complex64)

    print("Begin preparation")

    # time complexity 2**N * d = d**2
    for b in range(2**m):
        for y in range(2**(N-m)):
            for i in range(d):
                by = b*(2**(N-m)) + y 
                v1[b, i] += UTX[i, by] * UdaggerX[i, by]
        
    v2 = optimized_numba_sum(d, m, N, XUstar, XU, UTX, UdaggerX, v2)
    # time complexity d * d * d * (2**(N-m)) approx d**4

    # for i in range(d):
    #     for j in range(d):
    #         if i == j:
    #             continue
    #         for b in range(2**m):
    #             for x in range(2**(N-m)):
    #                 for y in range(2**(N-m)):
    #                     bx = b*(2**(N-m)) + x
    #                     by = b*(2**(N-m)) + y 
    #                     v2[i, j] += XUstar[bx, i] * XU[bx, j] * UTX[i, by] * UdaggerX[j, by]
    
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

    res /= tp.shape[0]
    return 1 - res.item() 

def process_unitary(U, m, ns, *, context):
    N = context.N
    d = 2**N
    if U.shape != (d, d):
        raise ValueError("U shape must match the context qubit count")
    if not 0 <= m < N:
        raise ValueError("m must satisfy 0 <= m < N")
    H5, mats, weights = context.H5, context.mats, context.weights
    # here U is a d x d matrix. It is ok for d to be 2**10.
    UT = U.transpose()
    Udagger = UT.conjugate()
    Ustar = U.conjugate()

    XUstar = H5 @ Ustar
    XU = H5 @ U
    UTX = UT @ H5
    UdaggerX = Udagger @ H5

    det_pr = np.zeros_like(ns, dtype=np.complex64)

    import time
    start = time.perf_counter()

    v1, v2 = calcV(XU, UTX, UdaggerX, XUstar, m, N)

    end = time.perf_counter()

    elapsed = end - start
    print("Elapsed:", elapsed, "seconds")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    v1_t = torch.from_numpy(v1).to(device)
    v2_t = torch.from_numpy(v2).to(device)
    skip_pr = 0
    skip_wt = int(N / 2)
    for i in tqdm(range(4**N)):
        wt = weights[i]
        # if wt >= skip_wt:
        #     skip_pr += ((ns/4)**wt) * ((1-3*ns/4)**(N-wt))
        #     continue
        P = mats[i].astype(np.complex64)
        tp = Udagger @ P @ U
        ret = detect(torch.from_numpy(tp).to(device), v1_t, v2_t)
        for i, noise in enumerate(ns):
            det_pr[i] += ((noise/4)**wt) * ((1-3*noise/4)**(N-wt)) * ret
            
    print(skip_wt, skip_pr)
    det_rate = np.real(det_pr / ((1-(1-(3*ns/4))**N)))
    fidelity = ((1-(3*ns/4))**N / (1 - det_pr))**(1/(N-m))
    # fidelity = (((1-(3*ns/4))**N / (1 - det_pr)) + 1) / 2

    # return det_pr, det_rate, fidelity
    return (1-det_pr)*(1-m/N), np.real(fidelity.item())
