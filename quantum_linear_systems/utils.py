"""Utility functions that can be imported by either implementation."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def make_matrix_hermitian(matrix):
    """Creates a hermitian version of a NxM :obj:np.array A as a  (N+M)x(N+M) block matrix [[0 ,A], [A_dagger, 0]]."""
    shape = matrix.shape
    upper_zero = np.zeros(shape=(shape[0], shape[0]))
    lower_zero = np.zeros(shape=(shape[1], shape[1]))
    matrix_dagger = matrix.conj().T
    hermitian_matrix = np.block([[upper_zero, matrix], [matrix_dagger, lower_zero]])
    assert np.array_equal(hermitian_matrix, hermitian_matrix.conj().T)
    return hermitian_matrix


def expand_b_vector(unexpanded_vector, non_hermitian_matrix):
    """Expand vector according to the expansion of the matrix to make it hermitian b -> (b 0)."""
    shape = non_hermitian_matrix.shape
    lower_zero = np.zeros(shape=(shape[1], 1))
    return np.block([[unexpanded_vector], [lower_zero]])


def extract_x_from_expanded(expanded_solution_vector: np.array, non_hermitian_matrix: np.array = None):
    """The expanded problem returns a vector y=(0 x), this function returns x from input y."""
    if non_hermitian_matrix is not None:
        index = non_hermitian_matrix.shape[0]
    else:
        index = int(expanded_solution_vector.flatten().shape[0] / 2)
    return expanded_solution_vector[index:].flatten()


def extract_hhl_solution_vector_from_state_vector(hermitian_matrix: np.array, state_vector: np.array):
    """Extract the solution vector x from the full state vector of the HHL problem which also includes 1 aux. qubit and
    multiple work qubits encoding the eigenvalues.
    """
    size_of_hermitian_matrix = hermitian_matrix.shape[1]
    number_of_qubits_in_result = int(np.log2(len(state_vector)))
    binary_rep = "1" + (number_of_qubits_in_result-1) * "0"
    not_normalized_vec = np.real(state_vector[int(binary_rep, 2):(int(binary_rep, 2) + size_of_hermitian_matrix)])

    return not_normalized_vec / np.linalg.norm(not_normalized_vec)


def plot_csol_vs_qsol(classical_solution, quantum_solution, title):
    matplotlib.use('Qt5Agg')
    plt.plot(classical_solution, "bo", label="classical")
    plt.plot(quantum_solution, "ro", label="HHL")
    plt.legend()
    plt.xlabel("$i$")
    plt.ylabel("$x_i$")
    plt.ylim(0, 1)
    plt.title(title)
    plt.show()
