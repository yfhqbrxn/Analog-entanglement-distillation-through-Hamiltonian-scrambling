import numpy as np
from qiskit.quantum_info import Pauli

# Single-qubit Paulis in IXYZ order for sampling.
_PAULI_SYMBOLS = np.array(["I", "X", "Y", "Z"])


def sample_pauli_iid(n, noise_strength, rng=None):
    """
    Sample an n-qubit Pauli string under local i.i.d. depolarizing noise.

    Each qubit is I with probability 1 - 3p/4 and X/Y/Z with probability p/4.
    Returns (label, matrix, weight), where weight counts non-identity factors.
    """
    if not (0.0 <= noise_strength <= 1.0):
        raise ValueError("noise_strength must be in [0, 1].")

    p_identity = 1.0 - 0.75 * noise_strength
    if p_identity < 0.0:
        raise ValueError("noise_strength too large: identity probability became negative.")

    if rng is None:
        rng = np.random.default_rng()

    probs = np.array(
        [p_identity, noise_strength / 4.0, noise_strength / 4.0, noise_strength / 4.0],
        dtype=float,
    )
    symbols = rng.choice(_PAULI_SYMBOLS, size=n, p=probs)
    label = "".join(symbols.tolist())
    matrix = Pauli(label).to_matrix()
    weight = (symbols != "I").sum()
    return label, matrix, weight

if __name__ == "__main__":
    for i in range(10):
        lb, mt, wt = sample_pauli_iid(n=10, noise_strength=0.2)
        print(lb, wt)
