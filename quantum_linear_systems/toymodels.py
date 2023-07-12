"""Implementations of toy-models that can be imported by the algorithms."""
import numpy as np
from linear_solvers import NumPyLinearSolver

from quantum_linear_systems.utils import make_matrix_hermitian, expand_b_vector


def qiskit_4qubit_example():
    """Reproduces the qiskit 4-qubit example from https://learn.qiskit.org/course/ch-applications/
    solving-linear-systems-of-equations-using-hhl-and-its-qiskit-implementation#example1

    Returns
    matrix_a =   1  -1/3
                -1/3 1
    vector_b = (1,0)
    solution_x = (1.125, 0.375)
    """
    matrix_a = np.array([[1, -1/3], [-1/3, 1]])
    vector_b = np.array([[1], [0]])
    solution_x = np.array([[1.125], [0.375]])

    return matrix_a, vector_b, solution_x, "qiskit_4qubit"


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


def volterra_problem(n):
    """
    Define the Volterra integral equation problem ready for solving.

    Parameters:
        n (int): The size of the problem, such that the total size will be 2**n.

    Returns:
        A tuple containing the problem matrix and vector.
    """
    # starting with simplified Volterra integral equation x(t) = 1 - I(x(s)ds)0->t
    N = 2 ** n
    delta_s = 1 / N

    alpha = delta_s / 2

    vec = np.ones((N, 1))

    # prepare matrix A and vector b to be used for HHL, expanding them to a hermitian form of A --> A_tilde*x=b_tilde
    mat = volterra_a_matrix(size=N, a=alpha)

    print("A =", mat, "\n")
    print("b =", vec)

    # expand
    a_tilde = make_matrix_hermitian(mat)
    b_tilde = expand_b_vector(vec, mat)

    print("A_tilde =", a_tilde, "\n")
    print("b_tilde =", b_tilde)

    classical_solution = NumPyLinearSolver().solve(mat, vec).state.flatten()    # qiksit
    # classical_solution = np.linalg.solve(matrix_a, vector_b)                  # numpy (classiq)

    return a_tilde, b_tilde, classical_solution, "Volterra"


def integro_differential_a_matrix(a_matrix, t_n):
    """Build a matrix of arbitrary size representing the integro-differential toy model."""
    delta_t = 1 / t_n
    alpha_n = a_matrix.shape[0]
    identity_block = np.identity(alpha_n)
    zero_block = np.zeros((alpha_n, alpha_n))
    off_diagonal_block = - np.identity(alpha_n) - delta_t * a_matrix
    generated_block = []
    for i in range(t_n):
        if i == 0:
            generated_block.append([np.block([identity_block] + [zero_block for _ in range(t_n - 1)])])
        else:
            generated_block.append([np.block([[zero_block for _ in range(i-1)] +
                                             [off_diagonal_block, identity_block] +
                                             [zero_block for _ in range(t_n - (i + 1))]])])
    return np.block(generated_block)


def decompose_into_unitaries(matrix):
    return


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    t_n = 4
    a = np.random.random((t_n, t_n))

    result = integro_differential_a_matrix(a, t_n)
    print(result)
