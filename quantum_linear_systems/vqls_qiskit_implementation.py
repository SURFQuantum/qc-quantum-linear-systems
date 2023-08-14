"""VQLS implementation using Qiskit and https://github.com/QuantumApplicationLab/vqls-prototype"""
import time
from typing import Tuple

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit.quantum_info import Statevector

from vqls_prototype import VQLS, VQLSLog

from quantum_linear_systems.toymodels import ClassiqDemoExample, ToyModel
from quantum_linear_systems.utils import extract_x_from_expanded, print_results


def qiskit_vqls_implementation(matrix_a: np.ndarray, vector_b: np.ndarray, ansatz: QuantumCircuit
                               ) -> Tuple[QuantumCircuit, np.array]:
    """Qiskit HHL implementation based on https://github.com/QuantumApplicationLab/vqls-prototype ."""

    log = VQLSLog([], [])
    estimator = Estimator()
    sampler = Sampler()
    vqls = VQLS(
        estimator,
        ansatz,
        COBYLA(maxiter=250, disp=True),
        sampler=sampler,
        callback=log.update,
    )
    opt = {"use_overlap_test": False, "use_local_cost_function": False}
    res = vqls.solve(matrix_a, vector_b, opt)

    vqls_circuit = res.state
    vqls_solution_vector = np.real(Statevector(res.state).data)
    return vqls_circuit, vqls_solution_vector


def qiskit_vqls(model: ToyModel, ansatz: QuantumCircuit, show_circuit: bool = False):
    """Full implementation unified between classiq and qiskit."""
    start_time = time.time()

    # solve HHL using qiskit
    vqls_circuit, vqls_solution_vector = qiskit_vqls_implementation(matrix_a=model.matrix_a, vector_b=model.vector_b,
                                                                    ansatz=ansatz)

    np.set_printoptions(precision=3, suppress=True)

    if show_circuit:
        vqls_circuit.draw()

    # remove zeros
    print("x quantum vs classical solution")
    if len(vqls_solution_vector) > len(model.classical_solution):
        quantum_solution = extract_x_from_expanded(vqls_solution_vector)
    else:
        quantum_solution = vqls_solution_vector
    # normalize
    quantum_solution /= np.linalg.norm(quantum_solution)

    qc_basis = vqls_circuit.decompose(reps=5)
    print(f"Comparing depths original {vqls_circuit.depth()} vs. decomposed {qc_basis.depth()}")

    return quantum_solution, model.classical_solution, qc_basis.depth(), vqls_circuit.width(), time.time() - start_time


if __name__ == "__main__":
    # starting with simplified Volterra integral equation x(t) = 1 - I(x(s)ds)0->t
    N = 2

    toymodel = ClassiqDemoExample()

    vqls_ansatz = RealAmplitudes(num_qubits=N, entanglement="full", reps=3, insert_barriers=False)

    toymodel = ClassiqDemoExample()

    qsol, csol, depth, width, run_time = qiskit_vqls(model=toymodel, show_circuit=True, ansatz=vqls_ansatz)

    print_results(quantum_solution=qsol, classical_solution=csol, run_time=run_time, name=toymodel.name, plot=True)