"""
Simple quantum simulation utilities for preparing multiple EPR pairs and
applying independent depolarizing noise to one subsystem of each pair.

This script uses Qiskit's statevector and density-matrix tools so that the
simulation stays lightweight (only 10 qubits for 5 EPR pairs) and quickly
produces fidelity diagnostics.
"""

from __future__ import annotations
from pathlib import Path
from tqdm import tqdm

from dataclasses import dataclass
from itertools import product
from typing import List, Mapping, Sequence, Tuple, Optional
from scipy.linalg import expm

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    DensityMatrix,
    Operator,
    SparsePauliOp,
    Statevector,
    state_fidelity,
    Kraus,
    partial_trace,
    Pauli
)

import matplotlib.pyplot as plt

@dataclass(frozen=True)
class SimulationResult:
    """Container for simulation outputs."""

    ideal_state: DensityMatrix
    noisy_state: DensityMatrix
    per_pair_fidelity: Tuple[float, ...]


PauliTerms = Mapping[str, complex] | Sequence[Tuple[str, complex]]
QuantumState = DensityMatrix | Statevector

def _epr_pair_circuit(num_pairs: int) -> QuantumCircuit:
    """Prepare |Phi+> EPR pairs on registers [0..n-1] and [n..2n-1]."""
    qc = QuantumCircuit(2 * num_pairs)
    for idx in range(num_pairs):
        qc.h(idx)
        qc.cx(idx, num_pairs + idx)
    return qc


def _bell_pair_density() -> DensityMatrix:
    """Return the 2-qubit density matrix of an ideal |Phi+> Bell pair."""
    bell_circuit = QuantumCircuit(2)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    return DensityMatrix.from_instruction(bell_circuit)


def _reduced_state(rho: DensityMatrix, keep_qubits: Sequence[int]) -> DensityMatrix:
    """Return the reduced density matrix whose support is `keep_qubits`."""
    if not keep_qubits:
        raise ValueError("keep_qubits must contain at least one index.")
    unique_keep = sorted(set(keep_qubits))

    total_qubits = rho.num_qubits
    out_of_range = [q for q in unique_keep if not (0 <= q < total_qubits)]
    if out_of_range:
        raise ValueError(f"Qubit indices out of range: {out_of_range}")

    trace_out = sorted([q for q in range(total_qubits) if q not in unique_keep], reverse=True)
    ret = partial_trace(rho, trace_out)
    # ret = rho
    # for q in trace_out:
    #     ret = partial_trace(ret, [q])   
    #     print(q, ret)     
    return ret

    # return rho.partial_trace(trace_out)
    # return partial_trace(rho, [trace_out])


def _single_qubit_depolarizing(prob: float) -> Kraus:
    """
    Build a 1-qubit depolarizing channel as Kraus operators.

    This is compatible with older Qiskit (e.g., 2.2.x) where DepolarizingChannel
    may not be importable directly.
    """
    if not (0.0 <= prob <= 1.0):
        raise ValueError("Depolarizing probability must lie in [0, 1].")
    p_identity = 1.0 - 3.0 * prob / 4.0
    p_error = prob / 4.0
    coeff_i = np.sqrt(p_identity)
    coeff_err = np.sqrt(p_error)
    I = coeff_i * np.eye(2, dtype=complex)
    X = coeff_err * np.array([[0, 1], [1, 0]], dtype=complex)
    Y = coeff_err * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = coeff_err * np.array([[1, 0], [0, -1]], dtype=complex)
    return Kraus([I, X, Y, Z])


def _projector_for_outcome(
    outcome_bits: Sequence[int], measured_qubits: Sequence[int], num_qubits: int
) -> np.ndarray:
    """Build the full-system projector for a computational-basis outcome."""
    measured_map = dict(zip(measured_qubits, outcome_bits))
    proj0 = np.array([[1, 0], [0, 0]], dtype=complex)
    proj1 = np.array([[0, 0], [0, 1]], dtype=complex)

    ops = []
    for q in range(num_qubits):
        if q in measured_map:
            ops.append(proj1 if measured_map[q] == 1 else proj0)
        else:
            ops.append(np.eye(2, dtype=complex))

    full_proj = ops[0]
    for op in ops[1:]:
        full_proj = np.kron(op, full_proj)
    return full_proj


def measure_and_postselect(
    state: QuantumState,
    measured_qubits: Sequence[int],
    outcome: Optional[Sequence[int]] = None,
    hadamard_basis: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[str, float, DensityMatrix]:
    """
    Perform measurement on selected qubits and return post-measurement state.

    Args:
        state: Input statevector or density matrix.
        measured_qubits: Indices of qubits to measure.
        outcome: Optional bit values to post-select. If None, an outcome is sampled
            according to the measurement probabilities.
        hadamard_basis: If True, measure in the Hadamard (X) basis; otherwise in
            computational (Z) basis.
        rng: Optional NumPy generator for reproducible sampling when outcome is None.

    Returns:
        (outcome_bitstring, probability, post_state), where post_state is a DensityMatrix
        over the unmeasured qubits.
    """
    if not measured_qubits:
        raise ValueError("measured_qubits must contain at least one qubit.")

    measured_unique = tuple(sorted(set(measured_qubits)))
    num_qubits = state.num_qubits
    if any(q < 0 or q >= num_qubits for q in measured_unique):
        raise ValueError("Measured qubits out of range for the given state.")

    rho = DensityMatrix(state)
    unmeasured = [q for q in range(num_qubits) if q not in measured_unique]

    # probs = []
    # outcomes = []
    # for bits in product([0, 1], repeat=len(measured_unique)):
    #     proj = _projector_for_outcome(bits, measured_unique, num_qubits)
    #     p = float(np.real_if_close(np.trace(proj @ rho.data)))
    #     probs.append(max(p, 0.0))
    #     outcomes.append(bits)

    # probs = np.array(probs, dtype=float)
    # probs = probs / probs.sum()

    if hadamard_basis:
        # Rotate measured qubits by H so computational-basis projection matches X-basis measurement.
        H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
        ops = []
        for q in range(num_qubits):
            ops.append(H if q in measured_unique else np.eye(2, dtype=complex))

        U = ops[0]
        for op in ops[1:]:
            U = np.kron(op, U)

        rho = DensityMatrix(U @ rho.data @ U.conj().T)

    if outcome is None:
        pass
        # rng = rng or np.random.default_rng()
        # index = rng.choice(len(outcomes), p=probs)
        # outcome_bits = outcomes[index]
    else:
        outcome_bits = tuple(int(b) for b in outcome)
        # if len(outcome_bits) != len(measured_unique):
        #     raise ValueError("Outcome length does not match number of measured qubits.")
        # try:
        #     index = outcomes.index(outcome_bits)
        # except ValueError as exc:
        #     raise ValueError("Invalid outcome bits for measured qubits.") from exc

    proj = _projector_for_outcome(outcome_bits, measured_unique, num_qubits)
    numer = proj @ rho.data @ proj
    prob = np.real_if_close(np.trace(numer))
    if prob <= 0:
        raise ValueError("Selected outcome has zero probability.")
    post_full = DensityMatrix(numer / prob)
    post_reduced = _reduced_state(post_full, unmeasured) if unmeasured else post_full

    bitstring = "".join(str(b) for b in outcome_bits)
    return bitstring, float(prob), post_reduced


def test() -> None:
    """
    Quick check for _reduced_state on a 3-qubit product state |0>|+>|->.
    Prints the single-qubit reduced density matrices for each position.
    """
    # |+> = (|0> + |1>)/sqrt(2), |-> = (|0> - |1>)/sqrt(2)
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    minus = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2.0)
    zero = np.array([1.0, 0.0], dtype=complex)

    state = np.kron(np.kron(zero, plus), minus)  # |0>|+>|->
    rho = DensityMatrix(state)

    for idx in range(3):
        reduced = _reduced_state(rho, [idx])
        print(f"Reduced state keeping qubit {idx}:\n{reduced.data}\n")


def test2() -> None:
    """
    Check measure_and_postselect on two Bell pairs.
    Prepare |Phi+>_{01} ⊗ |Phi+>_{23}, measure qubits [1, 2],
    and print the post-selected state on the remaining qubits.
    """
    bell = _bell_pair_density()
    rho = DensityMatrix(np.kron(bell.data, bell.data))  # 4-qubit state

    measured = [0, 2]
    outcome, prob, post = measure_and_postselect(rho, measured_qubits=measured, outcome=[1, 1])
    print(f"Post-selected outcome on qubits {measured}: {outcome} with probability {prob}")
    print(f"Post-measurement state (qubits [0, 3] remaining):\n{post.data}\n")

def _pauli_terms_to_sparse_op(pauli_terms: PauliTerms, num_qubits: int) -> SparsePauliOp:
    """Convert a collection of Pauli terms to SparsePauliOp."""
    if isinstance(pauli_terms, Mapping):
        items = pauli_terms.items()
    else:
        items = pauli_terms

    term_list: List[Tuple[str, complex]] = []
    for label, coeff in items:
        label = label.upper()
        if len(label) != num_qubits:
            raise ValueError(
                f"Pauli label '{label}' does not match {num_qubits}-qubit register."
            )
        term_list.append((label, complex(coeff)))

    if not term_list:
        raise ValueError("pauli_terms must contain at least one Pauli string.")

    return SparsePauliOp.from_list(term_list)


def _time_evolution_unitary(pauli_terms: PauliTerms, time: float, num_qubits: int) -> Operator:
    """
    Build exp(-i * H * time) for H expressed as a linear combination of Paulis.

    Returns:
        An Operator representing the time-evolution unitary on `num_qubits`.
    """
    # sparse_op = _pauli_terms_to_sparse_op(pauli_terms, num_qubits)
    H = _pauli_terms_to_sparse_op(pauli_terms, num_qubits)
    U = Operator(expm(-1j * time * H.to_matrix()))
    return U
    # scaled = sparse_op * (-time)
    # return scaled.exp_i()


def _time_evolution_unitary_from_matrix(H: np.ndarray, time: float) -> Operator:
    """
    Build exp(-i * H * time) when the Hamiltonian is already provided as a matrix.
    """
    return Operator(expm(-1j * time * H))


def _apply_local_hamiltonian(
    state: QuantumState,
    hamiltonian: PauliTerms | np.ndarray,
    time: float,
    target_qubits: Sequence[int],
    num_pairs: int,
) -> QuantumState:
    """Apply exp(-iHt) generated by either Pauli terms or a full matrix to the specified qubits."""
    expected_qubits = 2 * num_pairs
    if state.num_qubits != expected_qubits:
        raise ValueError(
            f"State has {state.num_qubits} qubits, expected {expected_qubits} "
            f"for {num_pairs} EPR pairs."
        )

    if isinstance(hamiltonian, np.ndarray):
        unitary = _time_evolution_unitary_from_matrix(hamiltonian, time)
    else:
        unitary = _time_evolution_unitary(hamiltonian, time, len(target_qubits))
    return state.evolve(unitary, qargs=list(target_qubits))


def evolve_first_half(
    state: QuantumState, hamiltonian: PauliTerms | np.ndarray, time: float, num_pairs: int = 5
) -> QuantumState:
    """
    Apply a 5-qubit Hamiltonian evolution to the first half (qubits 0..4).
    """
    targets = range(0, num_pairs)
    return _apply_local_hamiltonian(state, hamiltonian, time, targets, num_pairs)


def evolve_second_half(
    state: QuantumState, hamiltonian: PauliTerms | np.ndarray, time: float, num_pairs: int = 5
) -> QuantumState:
    """
    Apply a 5-qubit Hamiltonian evolution to the second half (qubits num_pairs..2*num_pairs-1).
    """
    targets = range(num_pairs, 2 * num_pairs)
    return _apply_local_hamiltonian(state, hamiltonian, time, targets, num_pairs)


def simulate_iid_depolarizing(
    depolarizing_prob: float,
    num_pairs: int = 5,
    affected_register: Sequence[int] | None = None,
) -> SimulationResult:
    """
    Create `num_pairs` EPR pairs and apply i.i.d. depolarizing noise on one side.

    Args:
        depolarizing_prob: Probability parameter p of the single-qubit depolarizing channel.
        num_pairs: Number of EPR pairs to prepare (default 5).
        affected_register: Iterable of qubit indices to receive the noise. If None,
            defaults to the "second" subsystem (indices num_pairs..2*num_pairs-1).

    Returns:
        SimulationResult with ideal/noisy density matrices and per-pair fidelities.
    """
    circuit = _epr_pair_circuit(num_pairs)
    ideal_state = DensityMatrix.from_instruction(circuit)
    channel = _single_qubit_depolarizing(depolarizing_prob)

    targets = (
        affected_register
        if affected_register is not None
        else range(num_pairs, 2 * num_pairs)
    )

    noisy_state = ideal_state
    for q in targets:
        noisy_state = noisy_state.evolve(channel, qargs=[q])

    bell_dm = _bell_pair_density()
    fidelities: List[float] = []
    for pair_idx in range(num_pairs):
        keep = [pair_idx, num_pairs + pair_idx]
        reduced = _reduced_state(noisy_state, keep)
        fidelities.append(float(state_fidelity(bell_dm, reduced)))

    return SimulationResult(
        ideal_state=ideal_state,
        noisy_state=noisy_state,
        per_pair_fidelity=tuple(fidelities),
    )

def IsingH(N):
    d = 2**N
    Ising = np.zeros((d, d), dtype=np.complex128)

    J0 = 1
    B0 = 2

    for i in range(N):
        j = (i+1) % N
        Ising += Pauli(''.join('X' if k in (i, j) else 'I' for k in range(N))).to_matrix() * J0
        Ising += Pauli(''.join('Z' if k in (i, ) else 'I' for k in range(N))).to_matrix() * B0

    return Ising

def TP(N):
    d = 2**N
    TP = np.zeros((d, d), dtype=np.complex64)
    alpha = 1

    J0 = 1
    B0 = 2

    for i in range(N):
        for j in range(i+1, N):
            TP += Pauli(''.join('X' if k in (i, j) else 'I' for k in range(N))).to_matrix() * J0 / (abs(j-i)**alpha)
        TP += Pauli(''.join('Z' if k in (i, ) else 'I' for k in range(N))).to_matrix() * B0 
    
    return TP

def RydbergH(N):
    d = 2**N
    # Rydberg
    Rydberg = np.zeros((d, d), dtype=np.complex128)

    Omega = 1.0
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

ts = np.linspace(0.05, 30, 50)

def simulateWholeProcess(N, H, result):
    fd_list = []
    yield_list = []
    for T in ts:
        # T = 2.5
        # ####################################################################
        evolved = evolve_first_half(result.noisy_state, H, time=T, num_pairs=N)
        evolved = evolve_second_half(evolved, H, time=-T, num_pairs=N)
        # overlap_with_ideal = state_fidelity(result.ideal_state, evolved)
        # print(overlap_with_ideal)

        m = 3
        measured = list(range(m)) + list(range(N, N+m))
        prb = 0
        weighted_fidelity = 0
        unmeasured = [q for q in range(2*N) if q not in measured]
        reduced_ideal = _reduced_state(result.ideal_state, unmeasured)
        for bits in tqdm(product([0, 1], repeat=m)):
            outcome = bits + bits
            _, prob, post = measure_and_postselect(evolved, measured_qubits=measured, outcome=list(outcome), hadamard_basis=True)
            prb += prob
            weighted_fidelity += prob * state_fidelity(post, reduced_ideal)
        # print(prb)
        # print((weighted_fidelity / prb)**(1/(N-m)))
        yield_list.append(prb * (N-m) / N)
        fd_list.append((weighted_fidelity / prb)**(1/(N-m)))
    return yield_list, fd_list


def main() -> None:
    # This test tells us that the qiskit has an abnormal order rather than common ordering
    # Like if we have 4-qubits, typically we would order it in the way of |q0,q1,q2,q3>. But the qiskit's way of ordering those qubits are |q3,q2,q1,q0>.

    N = 5

    """Example execution with five EPR pairs and 2% depolarizing noise."""
    result = simulate_iid_depolarizing(depolarizing_prob=0.2, num_pairs=N)
    avg_fidelity = sum(result.per_pair_fidelity) / len(result.per_pair_fidelity)

    print("Per-pair Bell fidelities:")
    for idx, fidelity in enumerate(result.per_pair_fidelity):
        print(f"  Pair {idx}: {fidelity:.6f}")
    print(f"Average fidelity: {avg_fidelity:.6f}")

    # Choose a Hamiltonian defined as a full matrix (e.g., Ising or Rydberg).
    # yields_Is, fds_Is = simulateWholeProcess(N, IsingH(N), result)
    yields_Is, fds_Is = simulateWholeProcess(N, TP(N), result)
    yields_Ryd, fds_Ryd = simulateWholeProcess(N, RydbergH(N), result)
    # ####################################################################

    fig, ax_left = plt.subplots(figsize=(7, 4))
    fid_is_ln, = ax_left.plot(ts, fds_Is, marker='o', color='C0', label='Trapped-ion fidelity')
    fid_ryd_ln, = ax_left.plot(ts, fds_Ryd, marker='o', color='C1', label='Rydberg fidelity')
    ax_left.set_xlabel('time T')
    ax_left.set_ylabel('fidelity')
    ax_left.set_ylim(top=1.00)
    ax_left.grid(alpha=0.3, linestyle='--', linewidth=0.6)

    ax_right = ax_left.twinx()
    yield_is_ln, = ax_right.plot(ts, yields_Is, marker='x', linestyle='--', color='C0', label='Trapped-ion yield')
    yield_ryd_ln, = ax_right.plot(ts, yields_Ryd, marker='x', linestyle='--', color='C1', label='Rydberg yield')
    ax_right.set_ylabel('yield')
    ax_right.set_ylim(top=0.32)
    ax_right.set_ylim(bottom=0.20)

    handles = [fid_is_ln, fid_ryd_ln, yield_is_ln, yield_ryd_ln]
    labels = [h.get_label() for h in handles]
    ax_left.legend(handles, labels, loc='upper right', frameon=True, facecolor='white')
    fig.tight_layout()
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "finite_time.svg", bbox_inches="tight", pad_inches=0.12)

    # evolved = evolve_first_half(result.noisy_state, H, time=0.1, num_pairs=5)
    # evolved = evolve_second_half(evolved, H, time=0.12, num_pairs=5)
    # overlap_with_noisy = state_fidelity(result.noisy_state, evolved)
    # print(f"Overlap between noisy and evolved state: {overlap_with_noisy:.6f}")


if __name__ == "__main__":
    main()
