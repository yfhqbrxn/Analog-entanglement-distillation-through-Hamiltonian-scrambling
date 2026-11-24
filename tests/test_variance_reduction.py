import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'simulations'))
import numpy as np
import torch
from sampled_edp import build_eigenbases, prepare_detection
from torch_edp import paulis, detect
from variance_reduction_sampling import weight_probabilities, allocate_samples, stratified_result


class StratificationTests(unittest.TestCase):
    def test_weights_and_allocation(self):
        masses = weight_probabilities(6, .2)
        self.assertAlmostEqual(sum(masses), 1.)
        counts = allocate_samples(masses, 2, 3600)
        self.assertEqual(sum(counts.values()), 3600)
        self.assertTrue(all(k >= 2 for k in counts.values()))
        self.assertAlmostEqual(masses[-1], .15**6)

    def test_exact_and_sampled(self):
        n, p = 3, .2
        labels, matrices = paulis(n)
        for name, U in build_eigenbases(n).items():
            for m in [1, 2]:
                with self.subTest(method=name, m=m):
                    dagger, v1, v2 = prepare_detection(U, m)
                    P = 0.
                    for label, matrix in zip(labels, matrices):
                        w = sum(c != 'I' for c in label)
                        a = 1-np.real(detect(torch.from_numpy(dagger @ matrix.astype(np.complex64) @ U), v1, v2))
                        P += (p/4)**w*(1-3*p/4)**(n-w)*a
                    exact, records = stratified_result(U, m, p, exact_weight=3)
                    self.assertAlmostEqual(exact['acceptance'], P, places=6)
                    self.assertEqual(exact['acceptance_se'], 0.)
                    self.assertEqual(exact['evaluations'], 4**n-1)
                    sample, _ = stratified_result(U, m, p, samples=12, exact_weight=1)
                    repeat, _ = stratified_result(U, m, p, samples=12, exact_weight=1)
                    self.assertEqual(sample, repeat)
                    self.assertLessEqual(sample['acceptance_low'], P+1e-6)
                    self.assertGreaterEqual(sample['acceptance_high'], P-1e-6)
                    self.assertLessEqual(sample['fidelity_low'], exact['fidelity']+1e-6)
                    self.assertGreaterEqual(sample['fidelity_high'], exact['fidelity']-1e-6)
                    zero, _ = stratified_result(U, m, 0)
                    self.assertEqual(zero['acceptance'], 1.)
                    self.assertEqual(zero['fidelity'], 1.)
                    self.assertEqual(zero['acceptance_se'], 0.)
