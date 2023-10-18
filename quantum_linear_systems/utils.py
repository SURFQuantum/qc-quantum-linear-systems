"""Utility functions that can be imported by either implementation."""
from typing import Any

import numpy as np
from qiskit import qasm3
from qiskit import QuantumCircuit


def circuit_to_qasm3(circuit: QuantumCircuit, filename: str) -> Any:
    qasm_content = qasm3.dumps(circuit=circuit)
    print(qasm_content)
    with open(filename, "w") as stream:
        qasm3.dump(circuit=circuit, stream=stream)
    return qasm_content


def make_matrix_hermitian(matrix: np.ndarray) -> np.ndarray:
    """Creates a hermitian version of a NxM :obj:np.array A as a  (N+M)x(N+M) block
    matrix [[0 ,A], [A_dagger, 0]]."""
    shape = matrix.shape
    upper_zero = np.zeros(shape=(shape[0], shape[0]))
    lower_zero = np.zeros(shape=(shape[1], shape[1]))
    matrix_dagger = matrix.conj().T
    hermitian_matrix = np.block([[upper_zero, matrix], [matrix_dagger, lower_zero]])
    assert np.array_equal(hermitian_matrix, hermitian_matrix.conj().T)
    return hermitian_matrix


def expand_b_vector(
    unexpanded_vector: np.ndarray, non_square_matrix: np.ndarray = None
) -> np.ndarray:
    """Expand vector according to the expansion of the matrix to make it hermitian b ->
    (b 0)."""
    if non_square_matrix is not None:
        expand_by = non_square_matrix.shape[1]
    else:
        expand_by = unexpanded_vector.shape[0]
    lower_zero = np.zeros(shape=(expand_by, 1))
    return np.block([[unexpanded_vector], [lower_zero]]).flatten()


def is_expanded(matrix_a: np.ndarray, vector_b: np.ndarray) -> bool:
    """Check if a vector is expanded, meaning that the second half of the vector
    contains only zeros."""

    def has_corners_zero(matrix: np.ndarray) -> bool:
        # Check if the matrix is square (i.e., number of rows == number of columns)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix not square")

        # Check if the top-left and bottom-right corners are both 0
        quarter_size = matrix.shape[0] // 2
        top_left_quarter = matrix[:quarter_size, :quarter_size]
        bottom_right_quarter = matrix[quarter_size:, quarter_size:]

        if np.any(top_left_quarter != 0) or np.any(bottom_right_quarter != 0):
            return False

        return True

    vector_expanded = all(element == 0 for element in vector_b[len(vector_b) // 2 :])
    matrix_expanded = has_corners_zero(matrix_a)

    return vector_expanded and matrix_expanded


def extract_x_from_expanded(expanded_solution_vector: np.ndarray) -> np.ndarray:
    """The expanded problem returns a vector y=(0 x), this function returns x from input
    y."""
    if isinstance(expanded_solution_vector, list):
        expanded_solution_vector = np.array(expanded_solution_vector)
    assert isinstance(expanded_solution_vector, np.ndarray)

    return expanded_solution_vector[len(expanded_solution_vector) // 2 :].flatten()


def extract_hhl_solution_vector_from_state_vector(
    hermitian_matrix: np.array, state_vector: np.array
) -> np.ndarray:
    """Extract the solution vector x from the full state vector of the HHL problem which
    also includes 1 aux.

    qubit and multiple work qubits encoding the eigenvalues.
    """
    size_of_hermitian_matrix = hermitian_matrix.shape[1]
    number_of_qubits_in_result = int(np.log2(len(state_vector)))
    binary_rep = "1" + (number_of_qubits_in_result - 1) * "0"
    not_normalized_vec = np.real(
        state_vector[
            int(binary_rep, 2) : (int(binary_rep, 2) + size_of_hermitian_matrix)
        ]
    )

    return not_normalized_vec / np.linalg.norm(not_normalized_vec)


def normalize_quantum_by_classical_solution(
    quantum_solution: np.ndarray, classical_solution: np.ndarray
) -> np.ndarray:
    """Normalize the quantum solution to the same norm as the classical solution."""
    return (quantum_solution / np.linalg.norm(quantum_solution)) * np.linalg.norm(
        classical_solution
    )


def relative_distance_quantum_classical_solution(
    quantum_solution: np.ndarray, classical_solution: np.ndarray
) -> float:
    """Calculate relative distance of quantum and classical solutions in percent."""
    if quantum_solution.shape != classical_solution.shape:
        raise ValueError(
            f"Can't compute relative distance. Shape of quantum solution {quantum_solution.shape} "
            f"different from classical {classical_solution.shape}."
        )
    return float(
        np.linalg.norm(classical_solution - quantum_solution)
        / np.linalg.norm(classical_solution)
        * 100
    )


def generate_random_vector(size: int, uniformity_level: float) -> np.ndarray:
    """Generate a random vector of a given size while varying the level of uniformity.

    Parameters:
    size (int): The size of the vector to be generated.
    uniformity_level (float): A value between 0 and 1 that controls the level of uniformity.
        - 0: Completely non-uniform (sampled from normal distribution).
        - 1: Completely uniform (sampled from uniform distribution).
        - Values in between: A mixture of uniform and non-uniform elements.

    Returns:
    numpy.ndarray: A random vector of the specified size with varying uniformity which is normalized to 1.

    Example:
    >>> generate_random_vector(10, 0.2)
    array([ 0.23465278, -0.34781082,  0.        ,  0.        ,  0.        ,
            0.        ,  0.94386038,  0.        ,  0.        ,  0.        ])

    >>> generate_random_vector(5, 1.0)
    array([0.53812673, 0.16123648, 0.97176192, 0.46867161, 0.86048439])

    >>> generate_random_vector(8, 0.8)
    array([ 0.        ,  0.        ,  0.68090503,  0.10261092, -0.37432812,
            0.        ,  0.        ,  0.        ])
    """
    if uniformity_level < 0 or uniformity_level > 1:
        raise ValueError("uniformity_level should be between 0 and 1")

    # Generate random values based on uniformity_level
    if uniformity_level == 0:
        # Completely non-uniform (e.g., Gaussian distribution)
        random_vector = np.random.randn(size, 1)
    elif uniformity_level == 1:
        # Completely uniform (e.g., uniform distribution)
        random_vector = np.random.rand(size, 1)
    else:
        # Generate a mixture of uniform and non-uniform values
        num_uniform = int(size * uniformity_level)
        num_non_uniform = size - num_uniform

        # Generate a partially uniform part (e.g., uniform distribution)
        uniform_part = np.random.rand(num_uniform, 1)

        # Generate a partially non-uniform part (e.g., Gaussian distribution)
        non_uniform_part = np.random.randn(num_non_uniform, 1)

        # Concatenate the two parts
        random_vector = np.concatenate((uniform_part, non_uniform_part))

        # Shuffle the vector to mix the uniform and non-uniform parts
        np.random.shuffle(random_vector)

    return random_vector / np.linalg.norm(random_vector)


def vector_uniformity_entropy(vector: np.ndarray) -> float:
    """Calculate the entropy of a vector to measure its uniformity.

    Parameters:
    vector (numpy.ndarray): The input vector for which the uniformity is to be measured.

    Returns:
    float: The entropy of the vector. Lower values indicate greater uniformity.
    """
    # Ensure the input vector is a NumPy array
    vector = np.array(vector)

    # Normalize the vector to sum to 1 (assuming it represents a probability distribution)
    normalized_vector = vector / np.sum(vector)

    # Calculate entropy
    entropy = -np.sum(
        normalized_vector * np.log2(normalized_vector + 1e-10)
    )  # Adding a small epsilon for numerical stability

    return float(entropy)


def generate_s_sparse_matrix(matrix_size: int, s_non_zero_entries: int) -> np.ndarray:
    """Generate a random s-sparse matrix of a given size.

    Parameters:
    matrix_size (int): The size of the square matrix. It determines the number of rows and columns.
    s_non_zero_entries (int): The maximum number of non-zero entries allowed in any row or column.

    Returns:
    numpy.ndarray: A random s-sparse matrix of size matrix_size x matrix_size.

    Definition of s-sparse:
    A matrix is s-sparse if it has at most 's' non-zero entries in any row or column.

    Example:
    >>> generate_s_sparse_matrix(5, 2)
    array([[0.        , 0.28197835, 0.1216284 , 0.        , 0.26790398],
           [0.        , 0.        , 0.6133386 , 0.        , 0.        ],
           [0.93237137, 0.        , 0.        , 0.44796184, 0.        ],
           [0.        , 0.        , 0.        , 0.89026758, 0.        ],
           [0.        , 0.17753217, 0.        , 0.        , 0.43770629]])
    """
    if s_non_zero_entries <= 0:
        raise ValueError("s must be a positive integer")

    if matrix_size <= 0:
        raise ValueError("matrix_size must be a positive integer")

    if s_non_zero_entries > matrix_size:
        raise ValueError("s cannot be greater than matrix_size")

    # Initialize an empty matrix with all zeros
    matrix = np.zeros((matrix_size, matrix_size), dtype=float)

    # Generate random non-zero values in the matrix
    for i in range(matrix_size):
        # Randomly choose 's' unique column indices for non-zero entries
        non_zero_columns = np.random.choice(
            matrix_size, s_non_zero_entries, replace=False
        )

        # Randomly assign non-zero values to these columns
        non_zero_values = np.random.rand(
            s_non_zero_entries
        )  # You can adjust this distribution as needed

        # Set the selected columns to the non-zero values
        matrix[i, non_zero_columns] = non_zero_values

    return matrix


def is_matrix_well_conditioned(matrix: np.ndarray, threshold: float = 10.0) -> bool:
    """Check if a matrix is well-conditioned based on a threshold.

    Parameters:
    matrix (numpy.ndarray): The matrix to be checked for well-conditioning.
    threshold (float, optional): Threshold to define well-conditioned-ness. Defaults to 10.

    Returns:
    bool: True if the matrix is well-conditioned, False otherwise.

    Definition of well-conditioned:
    A matrix is well-conditioned when its condition number is sufficiently close to 1.
    """
    return np.linalg.cond(matrix) <= threshold  # type: ignore[no-any-return]


# def is_matrix_well_conditioned(matrix, tolerance=1e-6):
#     """
#     Check if a matrix is well-conditioned based on its singular values.
#
#     Parameters:
#     matrix (numpy.ndarray): The matrix to be checked for well-conditioning.
#     tolerance (float, optional): A small positive number to handle numerical precision.
#         Default is 1e-6.
#
#     Returns:
#     bool: True if the matrix is well-conditioned, False otherwise.
#
#     Definition of well-conditioned:
#     A matrix is well-conditioned when its singular values lie between the reciprocal
#     of its condition number and 1, considering a small tolerance for numerical precision.
#
#     Example:
#     >>> A = np.array([[2.0, 1.0], [1.0, 2.0]])
#     >>> is_matrix_well_conditioned(A)
#     True
#
#     >>> B = np.array([[1e-6, 0], [0, 1e6]])
#     >>> is_matrix_well_conditioned(B)
#     False
#     """
#     # Compute the singular values of the matrix
#     # singular_values = np.linalg.svd(matrix, compute_uv=False)
#
#     # Calculate the condition number
#     # condition_number = singular_values[0] / singular_values[-1]
#     condition_number = np.linalg.cond(matrix)
#
#     # Define the lower and upper bounds for singular values
#     # lower_bound = 1.0 / (condition_number + tolerance)
#     # upper_bound = 1.0 + tolerance
#     # print(f"Lower bound {lower_bound}, upper bound {upper_bound}")
#
#     # Check if all singular values are within the bounds
#     # return all(lower_bound <= singular_values) and all(singular_values <= upper_bound)
#     return condition_number <= 10
