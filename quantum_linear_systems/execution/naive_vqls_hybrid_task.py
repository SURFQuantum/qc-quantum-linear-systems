"""This module implements a naive setup of a VQLS hybrid task where the quantum part is
saved as a qasm file and then submitted to the quantum hardware.

The results are then given to the classical part of the algorithm. This is looped over
in a for loop.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.primitives import Estimator
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.optimizers import SLSQP
from vqls_prototype import VQLS
from vqls_prototype import VQLSLog


def naive_hybrid_solve_vqls(
    matrix_a: np.ndarray,
    vector_b: np.ndarray,
    ansatz: QuantumCircuit = None,
    estimator: Estimator = Estimator(),
    optimizer_name: str = "cobyla",
    optimizer_max_iter: int = 250,
    show_circuit: bool = False,
) -> None:
    np.set_printoptions(precision=3, suppress=True)

    if ansatz is None:
        ansatz = RealAmplitudes(
            num_qubits=int(np.log2(matrix_a.shape[0])),
            entanglement="full",
            reps=3,
            insert_barriers=False,
        )
    if optimizer_name.lower() == "cobyla":
        optimizer = COBYLA(maxiter=optimizer_max_iter, disp=True)
    elif optimizer_name.lower() == "slsqp":
        optimizer = SLSQP(maxiter=optimizer_max_iter, disp=True)
    else:
        raise ValueError(f"Invalid optimizer_name: {optimizer_name}")

    if vector_b.ndim == 2:
        vector_b = vector_b.flatten()

    log = VQLSLog([], [])
    vqls = VQLS(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=Sampler(),
        callback=log.update,
    )
    opt = {"use_overlap_test": False, "use_local_cost_function": False}
    # todo: here we diverge and write our own function for solve, where we split quantum circuit and optimization
    vqls.solve(matrix_a, vector_b, opt)
