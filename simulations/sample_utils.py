"""Batched Pauli evaluation on CPU/CUDA, without dense Pauli matrices.

Qiskit labels are ordered most-significant qubit first. Preparation uses the
existing Numba kernels; only repeated Pauli evaluation is optimized here.
"""
from itertools import islice

import numpy as np
import torch
from sampled_edp import prepare_detection


def pauli_row_action(labels, n):
    """Return source rows and phases such that (P @ U)[r] = phase[r]*U[source[r]]."""
    rows = np.arange(2**n, dtype=np.int64)
    indices = np.empty((len(labels), len(rows)), dtype=np.int64)
    phases = np.empty(indices.shape, dtype=np.complex64)
    for j, label in enumerate(labels):
        if len(label) != n or any(c not in 'IXYZ' for c in label):
            raise ValueError('Expected an n-qubit IXYZ label')
        xmask = zmask = ny = 0
        for bit, letter in enumerate(reversed(label)):
            if letter in 'XY':
                xmask |= 1 << bit
            if letter in 'ZY':
                zmask |= 1 << bit
            ny += letter == 'Y'
        source = rows ^ xmask
        parity = np.zeros(len(rows), dtype=np.int8)
        for bit in range(n):
            if zmask & (1 << bit):
                parity ^= ((source >> bit) & 1).astype(np.int8)
        indices[j] = source
        phases[j] = (1j**ny) * (1-2*parity)
    return indices, phases


class BatchedPauliEvaluator:
    """Keep eigenbasis/kernels resident; share each transformed Pauli across m."""
    def __init__(self, U, measured, device='cpu', batch_size=8):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        d = U.shape[0]
        self.n = d.bit_length()-1
        if U.shape != (d, d) or d != 2**self.n:
            raise ValueError('Expected a square 2**n eigenbasis')
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.U = torch.as_tensor(np.ascontiguousarray(U), device=self.device)
        self.Udagger = self.U.conj().transpose(0, 1).contiguous()
        self.kernels = {}
        for m in sorted(set(measured)):
            _, v1, v2 = prepare_detection(U, m, device)
            # trace(v1 @ O @ v1.H) = sum_ij C_ij O_ij, without conjugating O.
            C = (v1.transpose(0, 1) @ v1.conj()).to(self.U.dtype)
            self.kernels[m] = (C, v2.to(self.U.dtype))
        self.transformations = 0

    @torch.no_grad()
    def transform(self, labels):
        indices, phases = pauli_row_action(labels, self.n)
        indices = torch.from_numpy(indices).to(self.device)
        phases = torch.from_numpy(phases).to(device=self.device, dtype=self.U.dtype)
        PU = self.U[indices] * phases.unsqueeze(-1)
        self.transformations += len(labels)
        return torch.matmul(self.Udagger, PU)

    @torch.no_grad()
    def acceptance(self, transformed, measured):
        # Transpose, not adjoint: retain the original detect() algebra.
        O = transformed * transformed.transpose(-2, -1)
        diagonal = transformed.diagonal(dim1=-2, dim2=-1)
        results = []
        for m in measured:
            C, v2 = self.kernels[m]
            first = (O*C).sum(dim=(-2, -1))
            second = ((diagonal @ v2)*diagonal).sum(dim=-1)
            results.append(((first+second)/self.U.shape[0]).real)
        # One device-to-host transfer per batch, not one synchronization per error.
        return torch.stack(results, dim=1).cpu().numpy()

    def evaluate(self, labels, measured=None, progress=None):
        """Return one row per label and one column per requested m; bounded scratch memory."""
        measured = list(self.kernels) if measured is None else list(measured)
        iterator = iter(labels)
        chunks = []
        while True:
            batch = list(islice(iterator, self.batch_size))
            if not batch:
                break
            values = self.acceptance(self.transform(batch), measured)
            if not np.all(np.isfinite(values)) or np.any(values < -1e-5) or np.any(values > 1+1e-5):
                raise ValueError('Conditional acceptance outside [0,1] or nonfinite')
            chunks.append(np.clip(values, 0., 1.))
            if progress is not None:
                progress(len(batch))
        return np.concatenate(chunks) if chunks else np.empty((0, len(measured)))
