"""Weight-stratified EDP: exact low weights, conditional sampling of higher weights."""
import argparse
import math
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from sampled_edp import build_eigenbases, plot_results
from sample_utils import BatchedPauliEvaluator


def weight_probabilities(n, p):
    return np.array([math.comb(n, w)*(3*p/4)**w*(1-3*p/4)**(n-w)
                     for w in range(n+1)])


def allocate_samples(probabilities, exact_weight, budget):
    """Fixed allocation, independent of observed outcomes; at least two per stratum."""
    if budget < 0:
        raise ValueError("Sampling budget cannot be negative")
    weights = [w for w in range(exact_weight+1, len(probabilities)) if probabilities[w] > 0]
    if not weights:
        return {}
    if budget < 2*len(weights):
        raise ValueError("Sampling budget must allow at least two draws per nonzero tail weight")
    masses = probabilities[weights]
    shares = (budget-2*len(weights))*masses/masses.sum()
    counts = np.floor(shares).astype(int)+2
    for index in np.argsort(-(shares-np.floor(shares)))[:budget-int(counts.sum())]:
        counts[index] += 1
    return dict(zip(weights, map(int, counts)))


def weight_labels(n, w):
    for positions in combinations(range(n), w):
        for letters in product("XYZ", repeat=w):
            label = ["I"]*n
            for pos, letter in zip(positions, letters):
                label[pos] = letter
            yield "".join(label)


def random_weight_label(n, w, rng):
    label = ["I"]*n
    for pos, letter in zip(rng.choice(n, size=w, replace=False), rng.choice(list("XYZ"), size=w)):
        label[pos] = letter
    return "".join(label)


def _summarize(records, n, m, p, confidence):
    acceptance = sum(r['contribution'] for r in records)
    sampled = [r for r in records if not r['exact']]
    variance = sum(r['probability']**2*r['variance']/r['evaluations'] for r in sampled)
    range_sum = sum(r['probability']**2/r['evaluations'] for r in sampled)
    radius = math.sqrt(math.log(2/(1-confidence))*range_sum/2)
    # Exact contributions are known; each remaining contribution lies in [0, probability].
    known = sum(r['contribution'] for r in records if r['exact'])
    low = max(known, acceptance-radius)
    high = min(known+sum(r['probability'] for r in sampled), acceptance+radius, 1.)
    qn = (1-3*p/4)**n
    F = (qn/acceptance)**(1/(n-m))
    se = math.sqrt(variance)
    result = dict(acceptance=acceptance, acceptance_se=se,
                  fidelity=F, fidelity_se=F*se/((n-m)*acceptance),
                  yield_value=acceptance*(n-m)/n, yield_se=se*(n-m)/n,
                  confidence=confidence, acceptance_low=low, acceptance_high=high,
                  fidelity_low=(qn/high)**(1/(n-m)), fidelity_high=(qn/low)**(1/(n-m)),
                  yield_low=low*(n-m)/n, yield_high=high*(n-m)/n,
                  evaluations=sum(r['evaluations'] for r in records))
    result['yield'] = result.pop('yield_value')
    return result


def stratified_results(U, measured, p, samples=3600, exact_weight=2, seed=23333,
                       device="cpu", confidence=.95, progress=False, batch_size=8):
    """Evaluate exact strata once for all m; retain independent sampled streams per m."""
    n = U.shape[0].bit_length()-1
    measured = sorted(set(measured))
    if not measured or any(m < 1 or m >= n for m in measured):
        raise ValueError("Require nonempty measured counts with 1 <= m < n")
    if not 0 <= p <= 1 or not 0 <= exact_weight <= n or not 0 < confidence < 1:
        raise ValueError("Invalid p, exact_weight, or confidence")
    masses = weight_probabilities(n, p)
    counts = allocate_samples(masses, exact_weight, samples)
    evaluator = BatchedPauliEvaluator(U, measured, device, batch_size)
    records = {m: [] for m in measured}
    total = 0
    for w, mass in enumerate(masses):
        if mass == 0 or w == 0:
            continue
        population = math.comb(n, w)*3**w
        exact = w <= exact_weight or counts.get(w, 0) >= population
        total += population if exact else counts[w]*len(measured)
    with tqdm(total=total, desc="Pauli transforms", unit="error", position=1,
              leave=False, disable=not progress, dynamic_ncols=True) as bar:
        for w, mass in enumerate(masses):
            population = math.comb(n, w)*3**w
            exact = w <= exact_weight or counts.get(w, 0) >= population
            if mass == 0 or w == 0:
                for m in measured:
                    records[m].append(dict(weight=w, probability=float(mass), population=population,
                        exact=True, evaluations=0, acceptance=1. if w == 0 else 0.,
                        variance=0., contribution=float(mass) if w == 0 else 0.))
                continue
            count = population if exact else counts[w]
            bar.set_description(f"Weight {w}: {'exact' if exact else 'sampled'}")
            if exact:
                # Each U.H @ P @ U is shared across every measurement count.
                values = evaluator.evaluate(weight_labels(n, w), measured, bar.update)
            for column, m in enumerate(measured):
                if exact:
                    data = values[:, column]
                else:
                    rng = np.random.default_rng(np.random.SeedSequence([seed, m, w]))
                    labels = (random_weight_label(n, w, rng) for _ in range(count))
                    data = evaluator.evaluate(labels, [m], bar.update)[:, 0]
                mean = float(data.mean(dtype=np.float64))
                variance = 0. if exact else float(data.astype(np.float64).var(ddof=1))
                records[m].append(dict(weight=w, probability=float(mass), population=population,
                    exact=exact, evaluations=count, acceptance=mean, variance=variance,
                    contribution=float(mass*mean)))
    return {m: (_summarize(records[m], n, m, p, confidence), records[m]) for m in measured}


def stratified_result(U, m, p, samples=3600, exact_weight=2, seed=23333,
                      device="cpu", confidence=.95, progress=False, batch_size=8):
    """Single-m interface. Bounds cover sampling error, not floating-point error."""
    return stratified_results(U, [m], p, samples, exact_weight, seed,
                              device, confidence, progress, batch_size)[m]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=6)
    parser.add_argument('--measured', nargs='+', type=int, default=[2, 3, 4])
    parser.add_argument('--p', type=float, default=.2)
    parser.add_argument('--samples', type=int, default=3600, help='Total tail budget per Hamiltonian/m, not per weight')
    parser.add_argument('--exact-weight', type=int, default=2)
    parser.add_argument('--seed', type=int, default=23333)
    parser.add_argument('--confidence', type=float, default=.95)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=8, help='Paulis per batch; lower to reduce memory')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    if args.n < 2 or any(m < 1 or m >= args.n for m in args.measured):
        parser.error('Require n >= 2 and 1 <= m < n')
    if not 0 <= args.p <= 1 or not 0 <= args.exact_weight <= args.n or not 0 < args.confidence < 1 or args.seed < 0:
        parser.error('Invalid probability, exact weight, confidence, or seed')
    try:
        counts = allocate_samples(weight_probabilities(args.n, args.p), args.exact_weight, args.samples)
    except ValueError as error:
        parser.error(str(error))
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu')
    print(f'Tail allocations (enumerated instead if cheaper): {counts}', flush=True)
    print('Building Hamiltonians...', flush=True)
    bases = build_eigenbases(args.n)
    out = Path(__file__).resolve().parents[1]/'outputs'
    out.mkdir(exist_ok=True)
    stem = f'fym_variance_reduced_n{args.n}_samples{args.samples}'
    rows = []
    with tqdm(total=len(bases), desc='Hamiltonians', unit='Hamiltonian') as sweep:
        for index, (method, U) in enumerate(bases.items()):
            sweep.set_description(method)
            results = stratified_results(U, args.measured, args.p, args.samples, args.exact_weight,
                                         args.seed+index, device, args.confidence, True, args.batch_size)
            for m, (result, _) in results.items():
                metadata = dict(method=method, n=args.n, m=m, p=args.p, samples_budget=args.samples,
                                exact_weight=args.exact_weight, seed=args.seed+index, device=device)
                rows.append(dict(**metadata, **result))
            sweep.update(1)
    plot_results(rows, out/f'{stem}.svg', args.p)
    print(f"Saved {out / (stem + '.svg')}")


if __name__ == '__main__':
    main()
