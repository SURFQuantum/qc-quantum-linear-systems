import numpy as np
from qiskit.quantum_info import Statevector
from linear_solvers import HHL, NumPyLinearSolver
from qiskit.algorithms.linear_solvers.hhl import HHL

def volterra_a_matrix(size, a):
    """Creates a matrix representing the linear system of the Volterra integral equation x(t) = 1 - INT(x(s)ds).
    Parameters
    ----------
    size : int
        size of the square matrix to be created.
    a : float
        alpha = delta S / 2 parametrization.
    Returns
    -------
    a_matrix : np.matrix
        size x size square matrix.
    """
    matrix_a = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                # diagonal first entry 1, rest (1 + a)
                if i == 0:
                    matrix_a[i, j] = 1
                else:
                    matrix_a[i, j] = 1 + a
            elif j == 0:
                # first column (except first entry which is covered above) is all a
                matrix_a[i, j] = a
            elif 0 < j < i:
                # rest of lower bottom triangle is 2a
                matrix_a[i, j] = 2 * a
    return np.matrix(matrix_a)


def make_matrix_hermitian(matrix):
    """Creates a hermitian version of a NxM :obj:np.matrix A as a  (N+M)x(N+M) block matrix [[0 ,A], [A_dagger, 0]]."""
    shape = matrix.shape
    upper_zero = np.matrix(np.zeros(shape))
    lower_zero = np.matrix(np.zeros(shape=(shape[1], shape[0])))
    matrix_dagger = matrix.getH()
    hermitian_matrix = np.matrix(np.block([[upper_zero, matrix], [matrix_dagger, lower_zero]]))
    # unittest, check if output is hermitian: A == A_dagger
    assert np.array_equal(hermitian_matrix, hermitian_matrix.getH())
    return hermitian_matrix


def expand_b_vector(unexpanded_vector, non_hermitian_matrix):
    """Expand vector according to the expansion of the matrix to make it hermitian b -> (b 0)."""
    shape = non_hermitian_matrix.shape
    lower_zero = np.matrix(np.zeros(shape=(shape[1], 1)))
    return np.block([[unexpanded_vector], [lower_zero]])


def extract_x_from_expanded(expanded_solution_vector, non_hermitian_matrix):
    """The expanded problem returns a vector y=(0 x), this function returns x from input y."""
    shape = non_hermitian_matrix.shape
    return np.array(expanded_solution_vector[shape[0]:])


def extract_hhl_solution_vector_from_state_vector(hermitian_matrix, state_vector):
    """Extract the solution vector x from the full state vector of the HHL problem which also includes 1 aux. qubit and
    multiple work qubits encoding the eigenvalues.
    """
    size_of_hermitian_matrix = hermitian_matrix.shape[1]
    number_of_qubits_in_result = int(np.log2(len(state_vector)))
    binary_rep = "1" + (number_of_qubits_in_result-1) * "0"
    return np.real(state_vector[int(binary_rep, 2):(int(binary_rep, 2) + size_of_hermitian_matrix)])


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

