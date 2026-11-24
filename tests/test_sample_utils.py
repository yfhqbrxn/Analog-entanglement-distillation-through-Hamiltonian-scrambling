"""Compare optimized Pauli action and contractions to the original dense kernels."""
import sys
import unittest
from itertools import product
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'simulations'))
import numpy as np
import torch
from qiskit.quantum_info import Pauli
from sampled_edp import prepare_detection
from torch_edp import detect
from sample_utils import BatchedPauliEvaluator
from variance_reduction_sampling import stratified_result, stratified_results


class BatchTests(unittest.TestCase):
    @staticmethod
    def eigenbasis():
        rng = np.random.default_rng(123)
        Z = rng.normal(size=(8, 8))+1j*rng.normal(size=(8, 8))
        return np.linalg.qr(Z)[0].astype(np.complex64)

    def check_dense(self, device):
        U = self.eigenbasis()
        labels = [''.join(x) for x in product('IXYZ', repeat=3)]
        evaluator = BatchedPauliEvaluator(U, [1, 2], device, batch_size=7)
        # Includes odd/even Y counts and both end-qubit positions.
        actual = evaluator.transform(labels).cpu().numpy()
        expected = np.array([U.conj().T @ Pauli(label).to_matrix().astype(np.complex64) @ U for label in labels])
        np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=2e-6)
        result = evaluator.evaluate(labels)
        for col, m in enumerate([1, 2]):
            _, v1, v2 = prepare_detection(U, m, device)
            original = [np.real(1-detect(torch.from_numpy(T).to(device), v1, v2)) for T in expected]
            np.testing.assert_allclose(result[:, col], original, atol=5e-7, rtol=2e-6)
        # Each Pauli was transformed once per pass, not separately for each m.
        self.assertEqual(evaluator.transformations, 2*len(labels))

    def test_cpu_dense(self):
        self.check_dense('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_dense(self):
        self.check_dense('cuda')

    def test_batches_and_shared_measurements(self):
        U = self.eigenbasis()
        together = stratified_results(U, [1, 2], .2, samples=12, exact_weight=1, batch_size=7)
        for m in [1, 2]:
            separate, _ = stratified_result(U, m, .2, samples=12, exact_weight=1, batch_size=1)
            for key, value in separate.items():
                self.assertAlmostEqual(together[m][0][key], value, places=6)
