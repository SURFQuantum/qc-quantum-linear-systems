from typing import Tuple

import numpy as np
import scipy
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import transpile
from qiskit.circuit.library import Initialize
from qiskit.circuit.library import PhaseEstimation as PhaseEstimation_QISKIT
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal
from qiskit.quantum_info import Statevector


def get_qiskit_hhl_results(
    a_matrix: np.ndarray,
    b_vector: np.ndarray,
    precision: int,
    optimization_level: int = 3,
) -> Tuple[float, float, float, float]:
    """This function creates an HHL circuit with qiskit, execute it and returns the
    depth, cx-counts and fidelity.

    Needs to adjusted to fit with other implementations.
    """
    num_qubits = int(np.log2(len(b_vector)))
    vector_circuit = QuantumCircuit(num_qubits)
    initi_vec = Initialize(b_vector / np.linalg.norm(b_vector))

    vector_circuit.append(initi_vec, list(range(num_qubits)))

    q = QuantumRegister(num_qubits, "q")
    unitary_qc = QuantumCircuit(q)
    exact_unitary = scipy.linalg.expm(1j * 2 * np.pi * a_matrix)
    unitary_mat = exact_unitary.tolist()
    unitary_qc.unitary(unitary_mat, q)
    qpe_qc = PhaseEstimation_QISKIT(precision, unitary_qc)
    reciprocal_circuit = ExactReciprocal(
        num_state_qubits=precision, scaling=1 / 2**precision
    )
    # Initialise the quantum registers
    qb = QuantumRegister(num_qubits)  # right hand side and solution
    ql = QuantumRegister(precision)  # eigenvalue evaluation qubits
    qf = QuantumRegister(1)  # flag qubits

    hhl_qc = QuantumCircuit(qb, ql, qf)

    # State preparation
    hhl_qc.append(vector_circuit, qb[:])
    # QPE
    hhl_qc.append(qpe_qc, ql[:] + qb[:])
    # Conditioned rotation
    hhl_qc.append(reciprocal_circuit, ql[::-1] + [qf[0]])

    # QPE inverse
    hhl_qc.append(qpe_qc.inverse(), ql[:] + qb[:])

    # transpile
    tqc = transpile(
        hhl_qc,
        basis_gates=["u3", "cx"],
        optimization_level=optimization_level,
    )
    depth = tqc.depth()
    cx_counts = tqc.count_ops()["cx"]
    total_q = tqc.width()

    # execute
    statevector = np.array(Statevector(tqc))

    # post_process
    all_entries = [np.binary_repr(k, total_q) for k in range(2**total_q)]
    sol_indices = [
        int(entry, 2)
        for entry in all_entries
        if entry[0] == "1" and entry[1 : precision + 1] == "0" * precision
    ]
    qsol = statevector[sol_indices] / (1 / 2**precision)

    sol_classical = np.linalg.solve(a_matrix, b_vector)
    fidelity = (
        np.abs(
            np.dot(
                sol_classical / np.linalg.norm(sol_classical),
                qsol / np.linalg.norm(qsol),
            )
        )
        ** 2
    )

    return total_q, depth, cx_counts, fidelity
