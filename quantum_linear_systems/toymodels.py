"""Implementations of toy-models that can be imported by the algorithms."""
import numpy as np

from quantum_linear_systems.utils import make_matrix_hermitian, expand_b_vector


class ToyModel:
    """
    A class representing a generic toy problem for linear systems.

    Parameters:
        name (str): The name or identifier for the toy problem.
        matrix (numpy.ndarray): The coefficient matrix for the linear system.
        vector (numpy.ndarray): The right-hand side vector for the linear system.
        csol (numpy.ndarray): The classical solution to the linear system.

    Attributes:
        name (str): The name or identifier for the toy problem.
        matrix_a (numpy.ndarray): The coefficient matrix for the linear system.
        vector_b (numpy.ndarray): The right-hand side vector for the linear system.
        classical_solution (numpy.ndarray): The classical solution to the linear system.
        num_qubits (int): The number of qubits involved.
    """
    def __init__(self, name: str, matrix: np.ndarray, vector: np.ndarray, csol: np.ndarray):
        if not isinstance(name, str):
            raise TypeError(f"Name of ToyModel must be of type str not {type(name)}.")
        for param in [matrix, vector, csol]:
            if not isinstance(param, np.ndarray):
                raise TypeError(f"Matrix, vector and csol of ToyModel must be of type np.ndarray not {type(param)}.")
        self.name = name
        self.matrix_a = matrix
        self.vector_b = vector
        self.classical_solution = csol.flatten()

    @property
    def num_qubits(self):
        """Return the number of qubits involved in the problem."""
        return int(np.log2(self.matrix_a.shape[0]))

    @staticmethod
    def classically_solve(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """
        Solve a linear system using classical methods.

        Parameters:
           mat (numpy.ndarray): Coefficient matrix of the linear system.
           vec (numpy.ndarray): Right-hand side vector of the linear system.

        Returns:
           numpy.ndarray: The solution vector of the linear system.
        """
        return np.linalg.solve(mat, vec)


class Qiskit4QubitExample(ToyModel):
    """
    Reproduces the qiskit "4-qubit-HHL" example from `https://learn.qiskit.org/course/ch-applications/
    solving-linear-systems-of-equations-using-hhl-and-its-qiskit-implementation#example1`

    matrix_a =   1  -1/3
                -1/3 1
    vector_b = (1,0)
    classical_solution = (1.125, 0.375)
    """
    def __init__(self):
        name = "Qiskit4QubitExample"
        matrix_a = np.array([[1, -1 / 3], [-1 / 3, 1]])
        vector_b = np.array([1, 0])
        classical_solution = np.array([[1.125], [0.375]])
        super().__init__(name=name, matrix=matrix_a, vector=vector_b, csol=classical_solution)


class VolterraProblem(ToyModel):
    """
    Define the Volterra integral equation problem ready for solving.

    Parameters:
        num_qubits (int): The size of the problem, such that the total size will be 2**num_qubits.

    """
    def __init__(self, num_qubits):
        name = f"VolterraProblem(n={num_qubits})"
        # starting with simplified Volterra integral equation x(t) = 1 - I(x(s)ds)0->t
        total_n = 2 ** num_qubits
        delta_s = 1 / total_n

        alpha = delta_s / 2

        vec = np.ones((total_n, 1))

        # prepare matrix A and vector b to be used for HHL
        # expanding them to a hermitian form of A --> A_tilde*x=b_tilde
        mat = self.volterra_a_matrix(size=total_n, alpha=alpha)

        print("A =", mat, "\n")
        print("b =", vec)

        # expand
        a_tilde = make_matrix_hermitian(mat)
        b_tilde = expand_b_vector(vec)

        print("A_tilde =", a_tilde, "\n")
        print("b_tilde =", b_tilde)

        classical_solution = self.classically_solve(mat, vec)

        super().__init__(name=name, matrix=a_tilde, vector=b_tilde, csol=classical_solution)

    @staticmethod
    def volterra_a_matrix(size, alpha):
        """Creates a matrix representing the linear system of the Volterra integral equation x(t) = 1 - INT(x(s)ds).
        Parameters
        ----------
        size : int
            size of the square matrix to be created.
        alpha : float
            alpha = delta S / 2 parametrization.
        Returns
        -------
        a_matrix : np.ndarray
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
                        matrix_a[i, j] = 1 + alpha
                elif j == 0:
                    # first column (except first entry which is covered above) is all a
                    matrix_a[i, j] = alpha
                elif 0 < j < i:
                    # rest of lower bottom triangle is 2a
                    matrix_a[i, j] = 2 * alpha
        return matrix_a


class ClassiqDemoExample(ToyModel):
    """
    Define the classiq demo problem. (See https://platform.classiq.io/advanced)

    """
    def __init__(self):
        name = "ClassiqDemoExample"
        matrix_a = np.array(
            [
                [0.28, -0.01, 0.02, -0.1],
                [-0.01, 0.5, -0.22, -0.07],
                [0.02, -0.22, 0.43, -0.05],
                [-0.1, -0.07, -0.05, 0.42],
            ]
        )

        vector_b = np.array([1, 2, 4, 3])
        vector_b = vector_b / np.linalg.norm(vector_b)

        classical_solution = self.classically_solve(matrix_a, vector_b)

        print("A =", matrix_a, "\n")
        print("b =", vector_b)
        super().__init__(name=name, matrix=matrix_a, vector=vector_b, csol=classical_solution)


class RandomNQubitProblem(ToyModel):
    """
    Define a problem consisting of a random MxM matrix and vector.

    Parameters:
        num_qubits (int): Number of qubits, such that the total size will be M=2**problem_size. Important: If the
        matrix is not hermitian and needs to be made hermitian the actual num_qubits will increase by 1 from the input.

    """
    def __init__(self, num_qubits):
        name = f"RandomNQubitProblem(N={num_qubits})"
        matrix_a = np.random.rand(2**num_qubits, 2**num_qubits)
        matrix_a += matrix_a.T  # make matrix symmetric (don't expand here, to keep problem size to N)
        vector_b = np.random.rand(2**num_qubits)

        classical_solution = self.classically_solve(matrix_a, vector_b)

        print("A =", matrix_a, "\n")
        print("b =", vector_b)
        super().__init__(name=name, matrix=matrix_a, vector=vector_b, csol=classical_solution)


def integro_differential_a_matrix(a_matrix: np.ndarray, time_discretization_steps: int):
    """Build a matrix of arbitrary size representing the integro-differential toy model."""
    delta_t = 1 / time_discretization_steps
    alpha_n = a_matrix.shape[0]
    identity_block = np.identity(alpha_n)
    zero_block = np.zeros((alpha_n, alpha_n))
    off_diagonal_block = - np.identity(alpha_n) - delta_t * a_matrix
    generated_block = []
    for i in range(time_discretization_steps):
        if i == 0:
            generated_block.append([np.block([identity_block] + [zero_block for _ in
                                                                 range(time_discretization_steps - 1)])])
        else:
            generated_block.append([np.block([[zero_block for _ in range(i-1)] +
                                             [off_diagonal_block, identity_block] +
                                             [zero_block for _ in range(time_discretization_steps - (i + 1))]])])
    return np.block(generated_block)


# def decompose_into_unitaries(matrix):
#     return


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    T_N = 4
    a = np.random.random((T_N, T_N))

    result = integro_differential_a_matrix(a, T_N)
    print(result)
