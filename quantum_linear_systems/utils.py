"""Utility functions that can be imported by either implementation."""
import numpy as np


def make_matrix_hermitian(matrix):
    """Creates a hermitian version of a NxM :obj:np.array A as a  (N+M)x(N+M) block matrix [[0 ,A], [A_dagger, 0]]."""
    shape = matrix.shape
    upper_zero = np.zeros(shape)
    lower_zero = np.zeros(shape=(shape[1], shape[0]))
    matrix_dagger = matrix.conj().T
    hermitian_matrix = np.block([[upper_zero, matrix], [matrix_dagger, lower_zero]])
    assert np.array_equal(hermitian_matrix, hermitian_matrix.conj().T)
    return hermitian_matrix


def expand_b_vector(unexpanded_vector, non_hermitian_matrix):
    """Expand vector according to the expansion of the matrix to make it hermitian b -> (b 0)."""
    shape = non_hermitian_matrix.shape
    lower_zero = np.zeros(shape=(shape[1], 1))
    return np.block([[unexpanded_vector], [lower_zero]])


def extract_x_from_expanded(expanded_solution_vector, non_hermitian_matrix):
    """The expanded problem returns a vector y=(0 x), this function returns x from input y."""
    shape = non_hermitian_matrix.shape
    return expanded_solution_vector[shape[0]:]


def extract_hhl_solution_vector_from_state_vector(hermitian_matrix, state_vector):
    """Extract the solution vector x from the full state vector of the HHL problem which also includes 1 aux. qubit and
    multiple work qubits encoding the eigenvalues.
    """
    size_of_hermitian_matrix = hermitian_matrix.shape[1]
    number_of_qubits_in_result = int(np.log2(len(state_vector)))
    binary_rep = "1" + (number_of_qubits_in_result-1) * "0"
    return np.real(state_vector[int(binary_rep, 2):(int(binary_rep, 2) + size_of_hermitian_matrix)])
