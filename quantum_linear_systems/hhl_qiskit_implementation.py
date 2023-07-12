import time
import numpy as np
from linear_solvers import HHL
from qiskit.quantum_info import Statevector

from quantum_linear_systems.toymodels import volterra_problem
from quantum_linear_systems.utils import extract_hhl_solution_vector_from_state_vector, \
    extract_x_from_expanded, plot_csol_vs_qsol


if __name__ == "__main__":
    start_time = time.time()
    # starting with simplified Volterra integral equation x(t) = 1 - I(x(s)ds)0->t
    n = 2
    A, b, csol, name = volterra_problem(n)

    # solve HHL using qiskit
    naive_hhl_solution = HHL().solve(matrix=A, vector=b)
    print(naive_hhl_solution.state)

    naive_state_vec = Statevector(naive_hhl_solution.state).data

    np.set_printoptions(precision=3, suppress=True)
    hhl_solution_vector = extract_hhl_solution_vector_from_state_vector(hermitian_matrix=A,
                                                                        state_vector=naive_state_vec)
    print("HHL solution", hhl_solution_vector)

    # attempt classical solution for comparison

    print('classical state solution:', csol)
    # remove zeros
    print("x quantum vs classical solution")
    qsol = extract_x_from_expanded(hhl_solution_vector)
    csol /= np.linalg.norm(csol)
    qsol /= np.linalg.norm(qsol)
    print("quantum", qsol)
    print("classical", csol)

    qc_original = naive_hhl_solution.state
    qc_basis = naive_hhl_solution.state.decompose(reps=5)

    plot_csol_vs_qsol(csol, qsol, "Qiskit")
    print(f"Comparing depths original {qc_original.depth()} vs. decomposed {qc_basis.depth()}")
    # print(qc_basis)
    print(f"Finished qiskit run in {time.time() - start_time}s.")

    if np.linalg.norm(csol - qsol) / np.linalg.norm(csol) > 0.2:
        raise RuntimeError("The HHL solution is too far from the classical one, please verify your algorithm.")
