"""VQLS implementation using Qiskit and https://github.com/QuantumApplicationLab/vqls-
prototype."""
import time
from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit.primitives import Estimator
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.optimizers import SLSQP
from vqls_prototype import VQLS

from quantum_linear_systems.plotting import print_results
from quantum_linear_systems.toymodels import ClassiqDemoExample
from quantum_linear_systems.utils import circuit_to_qasm3
from quantum_linear_systems.utils import extract_x_from_expanded
from quantum_linear_systems.utils import is_expanded


def solve_vqls_qiskit(
    matrix_a: np.ndarray,
    vector_b: np.ndarray,
    ansatz: QuantumCircuit = None,
    estimator: Estimator = Estimator(),
    optimizer_name: str = "cobyla",
    optimizer_max_iter: int = 250,
    show_circuit: bool = False,
) -> Tuple[np.ndarray, str, int, int, float]:
    """Qiskit HHL implementation based on https://github.com/QuantumApplicationLab/vqls-
    prototype ."""
    start_time = time.time()
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

    vqls = VQLS(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=Sampler(),
        options={"use_overlap_test": False, "use_local_cost_function": False},
    )
    res = vqls.solve(matrix_a, vector_b)

    vqls_circuit = res.state
    vqls_solution_vector = np.real(Statevector(res.state).data)

    quantum_solution = postprocess_solution(
        matrix_a=matrix_a, vector_b=vector_b, solution_x=vqls_solution_vector
    )

    qc_basis = vqls_circuit.decompose(reps=10)

    if show_circuit:
        print(qc_basis)

    # todo: fix, make sure this is the right circuit
    qasm_content = circuit_to_qasm3(
        circuit=vqls_circuit, filename="vqls_qiskit_circuit.qasm3"
    )

    print(
        f"Comparing depths original {vqls_circuit.depth()} vs. decomposed {qc_basis.depth()}"
    )

    return (
        quantum_solution,
        qasm_content,
        qc_basis.depth(),
        vqls_circuit.width(),
        time.time() - start_time,
    )


def postprocess_solution(
    matrix_a: np.ndarray, vector_b: np.ndarray, solution_x: np.ndarray
) -> np.ndarray:
    """Post-process the solution vector.

    This includes:
    - normalization (based on comparing the lhs and rhs of Ax=b)
    - sign-flipping
    - cutting zeros introduced when making matrix_a hermitian
    """
    # compare norm (Ax) to norm(b)
    lhs = np.matmul(matrix_a, solution_x)
    normalization = np.linalg.norm(vector_b) / np.linalg.norm(lhs)
    normalized_solution = solution_x * normalization

    # compare signs of lhs and rhs(necessary for vqls)
    for i, _ in enumerate(vector_b):
        if lhs.ndim == 1:
            same_sign = abs(vector_b[i]) + abs(lhs[i]) == abs(vector_b[i] + lhs[i])
        else:
            same_sign = abs(vector_b[i]) + abs(lhs[0, i]) == abs(
                vector_b[i] + lhs[0, i]
            )
        if not same_sign:
            break
    if not same_sign:
        normalized_solution = -normalized_solution

    # remove zeros
    if is_expanded(matrix_a, vector_b):
        normalized_solution = extract_x_from_expanded(normalized_solution)

    return normalized_solution


if __name__ == "__main__":
    N = 1

    model = ClassiqDemoExample()
    # model = HEPTrackReconstruction(num_detectors=5, num_particles=5)
    # runtimes(250): 3,3 =150s; 4,3=153s; 4,4=677s ;5,4=654s (c.25) ; 5,5=3492s (c0.34)
    # Note: neither memory nor cpu usage significant at these sizes
    # Note: after 250 iterations the cost is not low enough, would it make more sense to define different stop criteria
    qsol, _, depth, width, run_time = solve_vqls_qiskit(
        matrix_a=model.matrix_a, vector_b=model.vector_b, show_circuit=True
    )

    print_results(
        quantum_solution=qsol,
        classical_solution=model.classical_solution,
        run_time=run_time,
        name=model.name,
        plot=True,
    )
