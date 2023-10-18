"""HHL implementation using Qiskit."""
import time
from typing import Tuple

import numpy as np
from linear_solvers import HHL
from linear_solvers import LinearSolverResult
from qiskit.quantum_info import Statevector

from quantum_linear_systems.plotting import print_results
from quantum_linear_systems.toymodels import ClassiqDemoExample
from quantum_linear_systems.utils import circuit_to_qasm3
from quantum_linear_systems.utils import extract_hhl_solution_vector_from_state_vector
from quantum_linear_systems.utils import extract_x_from_expanded
from quantum_linear_systems.utils import is_expanded


def solve_hhl_qiskit(
    matrix_a: np.ndarray, vector_b: np.ndarray, show_circuit: bool = False
) -> Tuple[np.ndarray, str, int, int, float]:
    """Solve linear system Ax=b using HHL implemented in qiskit based on the quantum
    linear solvers package.

    See: https://github.com/anedumla/quantum_linear_solvers.git
    """
    np.set_printoptions(precision=3, suppress=True)
    start_time = time.time()

    # solve HHL using qiskit
    hhl_implementation = HHL()
    naive_hhl_solution: LinearSolverResult = hhl_implementation.solve(
        matrix=matrix_a, vector=vector_b
    )

    hhl_circuit = naive_hhl_solution.state
    # Get the value of nl
    qpe_register_size = hhl_circuit.qregs[
        1
    ].size  # qiskit calculates this itself from the matrix (hhl.py line 391)
    print(
        f"Size of solution register is {hhl_circuit.qregs[0].size} , QPE registers is {qpe_register_size}."
    )

    naive_state_vec = Statevector(hhl_circuit).data
    hhl_solution_vector = extract_hhl_solution_vector_from_state_vector(
        hermitian_matrix=matrix_a, state_vector=naive_state_vec
    )
    # scale normalized solution vector to norm of final state
    hhl_solution_vector = naive_hhl_solution.euclidean_norm * hhl_solution_vector

    if show_circuit:
        hhl_circuit.draw()

    # remove zeros
    print("x quantum vs classical solution")
    if is_expanded(matrix_a, vector_b):
        hhl_solution_vector = extract_x_from_expanded(hhl_solution_vector)

    qc_basis = hhl_circuit.decompose(reps=10)
    print(
        f"Comparing depths original {hhl_circuit.depth()} vs. decomposed {qc_basis.depth()}"
    )

    qasm_content = circuit_to_qasm3(
        circuit=qc_basis, filename="hhl_qiskit_circuit.qasm3"
    )

    return (
        hhl_solution_vector,
        qasm_content,
        qc_basis.depth(),
        hhl_circuit.width(),
        time.time() - start_time,
    )


if __name__ == "__main__":
    # starting with simplified Volterra integral equation x(t) = 1 - I(x(s)ds)0->t
    N = 2

    model = ClassiqDemoExample()

    qsol, _, depth, width, run_time = solve_hhl_qiskit(
        matrix_a=model.matrix_a, vector_b=model.vector_b, show_circuit=True
    )

    print_results(
        quantum_solution=qsol,
        classical_solution=model.classical_solution,
        run_time=run_time,
        name=model.name,
        plot=True,
    )
