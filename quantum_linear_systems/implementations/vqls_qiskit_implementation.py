"""VQLS implementation using Qiskit and https://github.com/QuantumApplicationLab/vqls-prototype"""
import time

import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import COBYLA

from vqls_prototype import VQLS, VQLSLog

from quantum_linear_systems.toymodels import ClassiqDemoExample
from quantum_linear_systems.utils import extract_x_from_expanded, is_expanded
from quantum_linear_systems.plotting import print_results


def solve_vqls_qiskit(matrix_a: np.ndarray, vector_b: np.ndarray, ansatz: QuantumCircuit, csol: np.ndarray,
                      show_circuit: bool = False):
    """Qiskit HHL implementation based on https://github.com/QuantumApplicationLab/vqls-prototype ."""
    # flatten vector such that qiskit doesn't bug out in state preparation
    start_time = time.time()
    np.set_printoptions(precision=3, suppress=True)

    if vector_b.ndim == 2:
        vector_b = vector_b.flatten()

    log = VQLSLog([], [])
    vqls = VQLS(
        Estimator(),
        ansatz,
        optimizer=COBYLA(maxiter=250, disp=True),
        sampler=Sampler(),
        callback=log.update,
    )
    opt = {"use_overlap_test": False, "use_local_cost_function": False}
    res = vqls.solve(matrix_a, vector_b, opt)

    vqls_circuit = res.state
    vqls_solution_vector = np.real(Statevector(res.state).data)

    # remove zeros
    print("x quantum vs classical solution")
    if is_expanded(matrix_a, vector_b):
        vqls_solution_vector = extract_x_from_expanded(vqls_solution_vector)

    # ensure we have the positive vector
    if np.sum(vqls_solution_vector) < 0:
        vqls_solution_vector = -vqls_solution_vector

    # todo: hack to have quantum and classical solution have the same norm
    quantum_solution = vqls_solution_vector * np.linalg.norm(csol)

    qc_basis = vqls_circuit.decompose(reps=10)

    if show_circuit:
        print(qc_basis)

    # todo: fix, make sure this is the right circuit
    qasm_content = vqls_circuit.qasm()

    print(f"Comparing depths original {vqls_circuit.depth()} vs. decomposed {qc_basis.depth()}")

    return quantum_solution, qasm_content, qc_basis.depth(), vqls_circuit.width(), time.time() - start_time


if __name__ == "__main__":
    N = 1

    model = ClassiqDemoExample()

    vqls_ansatz = RealAmplitudes(num_qubits=model.num_qubits,
                                 entanglement="full", reps=3, insert_barriers=False)

    qsol, _, depth, width, run_time = solve_vqls_qiskit(matrix_a=model.matrix_a, vector_b=model.vector_b,
                                                        show_circuit=True, ansatz=vqls_ansatz,
                                                        csol=model.classical_solution)

    print_results(quantum_solution=qsol, classical_solution=model.classical_solution,
                  run_time=run_time, name=model.name, plot=True)
