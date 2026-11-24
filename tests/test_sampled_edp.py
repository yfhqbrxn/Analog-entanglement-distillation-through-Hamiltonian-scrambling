"""Small-system numerical checks; run with python -m unittest discover -s tests."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulations"))
import numpy as np
import torch
from sampled_edp import build_eigenbases, prepare_detection, sampled_result
from torch_edp import paulis, detect


class SamplingTests(unittest.TestCase):
    def test_seed_reproducibility_and_zero_noise(self):
        U = build_eigenbases(3)["Rydberg"]
        a = sampled_result(U, 1, .2, 128, np.random.default_rng(123))
        b = sampled_result(U, 1, .2, 128, np.random.default_rng(123))
        self.assertEqual(a, b)
        for n in (2, 3):
            for method, U in build_eigenbases(n).items():
                with self.subTest(n=n, method=method):
                    result = sampled_result(U, 1, 0, 8, np.random.default_rng(1))
                    self.assertAlmostEqual(result["acceptance"], 1, places=5)
                    self.assertAlmostEqual(result["fidelity"], 1, places=5)
                    self.assertAlmostEqual(result["yield"], (n-1)/n, places=5)

    def test_sampling_against_exact_enumeration(self):
        n, p, count = 3, .2, 4096
        labels, matrices = paulis(n)
        for method, U in build_eigenbases(n).items():
            for m in (1, 2):
                with self.subTest(method=method, m=m):
                    Udagger, v1, v2 = prepare_detection(U, m)
                    rejection = 0.
                    for label, matrix in zip(labels, matrices):
                        weight = sum(c != 'I' for c in label)
                        probability = (p/4)**weight * (1-3*p/4)**(n-weight)
                        transformed = Udagger @ matrix.astype(np.complex64) @ U
                        rejection += probability * np.real(detect(torch.from_numpy(transformed), v1, v2))
                    acceptance = 1-rejection
                    exact_fidelity = ((1-3*p/4)**n / acceptance)**(1/(n-m))
                    result = sampled_result(U, m, p, count, np.random.default_rng(1234))
                    self.assertLess(abs(result['acceptance']-acceptance), 6*result['acceptance_se']+1e-6)
                    self.assertLess(abs(result['fidelity']-exact_fidelity), 6*result['fidelity_se']+1e-6)
                    self.assertLess(abs(result['yield']-acceptance*(n-m)/n), 6*result['yield_se']+1e-6)


if __name__ == '__main__':
    unittest.main()
