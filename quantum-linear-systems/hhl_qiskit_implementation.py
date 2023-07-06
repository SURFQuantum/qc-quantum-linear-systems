import numpy as np
from qiskit.quantum_info import Statevector
from linear_solvers import HHL, NumPyLinearSolver
# from qiskit.algorithms.linear_solvers.hhl import HHL
from toymodels import volterra_a_matrix
from utils import make_matrix_hermitian, expand_b_vector, extract_hhl_solution_vector_from_state_vector,\
    extract_x_from_expanded


if __name__ == "__main__":
    # starting with simplified Volterra integral equation x(t) = 1 - I(x(s)ds)0->t
    n = 2
    N = 2**n
    delta_s = 1/N      # TODO: What makes sense here?

    alpha = delta_s / 2

    b_vector = np.ones((N, 1))

    # prepare matrix A and vector b to be used for HHL, expanding them to a hermitian form of A --> A_tilde*x=b_tilde
    A = volterra_a_matrix(N, delta_s)
    print(A)
    A_tilde = make_matrix_hermitian(A)
    b_tilde = expand_b_vector(b_vector, A)

    # solve HHL using qiskit
    naive_hhl_solution = HHL().solve(matrix=A_tilde, vector=b_tilde)
    print(naive_hhl_solution.state)

    naive_state_vec = Statevector(naive_hhl_solution.state).data

    np.set_printoptions(precision=3, suppress=True)
    hhl_solution_vector = extract_hhl_solution_vector_from_state_vector(hermitian_matrix=A_tilde,
                                                                        state_vector=naive_state_vec)
    print("HHL solution", hhl_solution_vector)

    # attempt classical solution for comparison
    classical_solution = NumPyLinearSolver().solve(A_tilde, b_tilde / np.linalg.norm(b_tilde))
    print('classical state solution:', classical_solution.state)
    # remove zeros
    print("x quantum vs classical solution")
    print(extract_x_from_expanded(expanded_solution_vector=hhl_solution_vector, non_hermitian_matrix=A))
    print(extract_x_from_expanded(classical_solution.state, A))

    qc_original = naive_hhl_solution.state
    qc_basis = naive_hhl_solution.state.decompose(reps=5)
    print(f"Comparing depths original {qc_original.depth()} vs. decomposed {qc_basis.depth()}")
    # print(qc_basis)
