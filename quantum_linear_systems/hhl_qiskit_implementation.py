"""HHL implementation using Qiskit."""
import time

import numpy as np
from linear_solvers import HHL
from qiskit.quantum_info import Statevector

from quantum_linear_systems.toymodels import ToyModel, ClassiqDemoExample, Qiskit4QubitExample
from quantum_linear_systems.utils import (extract_hhl_solution_vector_from_state_vector,
                                          extract_x_from_expanded,
                                          print_results)


def qiskit_hhl_implementation(matrix_a: np.ndarray, vector_b: np.ndarray):
    """Qiskit HHL implementation based on quantum linear systems package."""
    hhl_implementation = HHL()
    naive_hhl_solution = hhl_implementation.solve(matrix=matrix_a, vector=vector_b)

    hhl_circuit = naive_hhl_solution.state
    # Get the value of nl
    qpe_register_size = hhl_circuit.qregs[1].size   # qiskit calculates this itself from the matrix (hhl.py line 391)
    print(f"Size of solution register is {hhl_circuit.qregs[0].size} , QPE registers is {qpe_register_size}.")

    naive_state_vec = Statevector(hhl_circuit).data
    hhl_solution_vector = extract_hhl_solution_vector_from_state_vector(hermitian_matrix=matrix_a,
                                                                        state_vector=naive_state_vec)
    # scale normalized solution vector to norm of final state
    hhl_solution_vector = naive_hhl_solution.euclidean_norm * hhl_solution_vector

    return hhl_circuit, hhl_solution_vector


def qiskit_hhl(model: ToyModel, show_circuit: bool = False):
    """Full implementation unified between classiq and qiskit."""
    print(f"Qiskit HHL solving {model.name}.")
    start_time = time.time()

    # solve HHL using qiskit
    hhl_circuit, hhl_solution_vector = qiskit_hhl_implementation(matrix_a=model.matrix_a, vector_b=model.vector_b)

    np.set_printoptions(precision=3, suppress=True)

    if show_circuit:
        hhl_circuit.draw()

    # remove zeros
    print("x quantum vs classical solution")
    if len(hhl_solution_vector) > len(model.classical_solution):
        quantum_solution = extract_x_from_expanded(hhl_solution_vector)
    else:
        quantum_solution = hhl_solution_vector

    qc_basis = hhl_circuit.decompose(reps=5)
    print(f"Comparing depths original {hhl_circuit.depth()} vs. decomposed {qc_basis.depth()}")

    return quantum_solution, model.classical_solution, qc_basis.depth(), hhl_circuit.width(), time.time() - start_time


if __name__ == "__main__":
    # starting with simplified Volterra integral equation x(t) = 1 - I(x(s)ds)0->t
    N = 2

    toymodel = ClassiqDemoExample()
    toymodel = Qiskit4QubitExample()

    qsol, csol, depth, width, run_time = qiskit_hhl(model=toymodel, show_circuit=True)

    print_results(quantum_solution=qsol, classical_solution=csol, run_time=run_time, name=toymodel.name, plot=True)
