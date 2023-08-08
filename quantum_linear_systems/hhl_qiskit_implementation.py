import time
import numpy as np
from linear_solvers import HHL
from qiskit.quantum_info import Statevector

from quantum_linear_systems.toymodels import classiq_demo_problem
from quantum_linear_systems.utils import extract_hhl_solution_vector_from_state_vector, \
    extract_x_from_expanded, plot_csol_vs_qsol


def qiskit_hhl_implementation(matrix_a, vector_b, precision=None):
    naive_hhl_solution = HHL().solve(matrix=matrix_a, vector=vector_b)

    hhl_circuit = naive_hhl_solution.state

    naive_state_vec = Statevector(hhl_circuit).data
    hhl_solution_vector = extract_hhl_solution_vector_from_state_vector(hermitian_matrix=matrix_a,
                                                                        state_vector=naive_state_vec)

    return hhl_circuit, hhl_solution_vector


if __name__ == "__main__":
    start_time = time.time()
    # starting with simplified Volterra integral equation x(t) = 1 - I(x(s)ds)0->t
    n = 2
    # A, b, csol, name = volterra_problem(n)
    A, b, csol, name = classiq_demo_problem()

    # solve HHL using qiskit
    hhl_circuit, hhl_solution_vector = qiskit_hhl_implementation(matrix_a=A, vector_b=b)

    np.set_printoptions(precision=3, suppress=True)
    print("HHL solution", hhl_solution_vector)

    # attempt classical solution for comparison

    print('classical state solution:', csol)
    # remove zeros
    print("x quantum vs classical solution")
    if len(hhl_solution_vector) > len(csol):
        qsol = extract_x_from_expanded(hhl_solution_vector)
    else:
        qsol = hhl_solution_vector
    csol /= np.linalg.norm(csol)
    qsol /= np.linalg.norm(qsol)
    print("quantum", qsol)
    print("classical", csol)

    qc_basis = hhl_circuit.decompose(reps=5)

    plot_csol_vs_qsol(csol, qsol, "Qiskit")
    print(f"Comparing depths original {hhl_circuit.depth()} vs. decomposed {qc_basis.depth()}")
    # print(qc_basis)
    print(f"Finished qiskit run in {time.time() - start_time}s.")

    if np.linalg.norm(csol - qsol) / np.linalg.norm(csol) > 0.2:
        raise RuntimeError("The HHL solution is too far from the classical one, please verify your algorithm.")
