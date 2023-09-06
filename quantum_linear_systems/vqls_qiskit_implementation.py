"""VQLS implementation using Qiskit and https://github.com/QuantumApplicationLab/vqls-prototype"""
import time
from typing import Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import COBYLA

from vqls_prototype import VQLS, VQLSLog

from quantum_linear_systems.toymodels import ClassiqDemoExample, ToyModel
from quantum_linear_systems.utils import extract_x_from_expanded, normalize_quantum_by_classical_solution
from quantum_linear_systems.plotting import print_results


def qiskit_vqls_implementation(matrix_a: np.ndarray, vector_b: np.ndarray, ansatz: QuantumCircuit
                               ) -> Tuple[QuantumCircuit, np.array]:
    """Qiskit HHL implementation based on https://github.com/QuantumApplicationLab/vqls-prototype ."""
    # flatten vector such that qiskit doesn't bug out in state preparation
    if vector_b.ndim == 2:
        vector_b = vector_b.flatten()

    log = VQLSLog([], [])
    estimator = Estimator()
    sampler = Sampler()
    vqls = VQLS(
        estimator,
        ansatz,
        optimizer=COBYLA(maxiter=250, disp=True),
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
    print(f"Qiskit VQLS solving {model.name}.")
    start_time = time.time()

    # solve VQLS using qiskit
    vqls_circuit, vqls_solution_vector = qiskit_vqls_implementation(matrix_a=model.matrix_a, vector_b=model.vector_b,
                                                                    ansatz=ansatz)

    np.set_printoptions(precision=3, suppress=True)

    if show_circuit:
        vqls_circuit.draw()

    # remove zeros
    print("x quantum vs classical solution")
    quantum_solution = extract_x_from_expanded(vqls_solution_vector)

    # ensure we have the positive vector
    if np.sum(quantum_solution) < 0:
        quantum_solution = -quantum_solution

    # todo: hack to have quantum and classical solution have the same norm
    quantum_solution = normalize_quantum_by_classical_solution(quantum_solution, model.classical_solution)

    qc_basis = vqls_circuit.decompose(reps=5)
    print(f"Comparing depths original {vqls_circuit.depth()} vs. decomposed {qc_basis.depth()}")

    return quantum_solution, model.classical_solution, qc_basis.depth(), vqls_circuit.width(), time.time() - start_time


if __name__ == "__main__":
    N = 1

    toymodel = ClassiqDemoExample()

    vqls_ansatz = RealAmplitudes(num_qubits=toymodel.num_qubits,
                                 entanglement="full", reps=3, insert_barriers=False)

    qsol, csol, depth, width, run_time = qiskit_vqls(model=toymodel, show_circuit=True, ansatz=vqls_ansatz)

    print_results(quantum_solution=qsol, classical_solution=csol, run_time=run_time, name=toymodel.name, plot=True)
