"""Implementations of toy-models that can be imported by the algorithms."""
import numpy as np

from quantum_linear_systems.utils import expand_b_vector
from quantum_linear_systems.utils import extract_x_from_expanded
from quantum_linear_systems.utils import generate_random_vector
from quantum_linear_systems.utils import generate_s_sparse_matrix
from quantum_linear_systems.utils import is_matrix_well_conditioned
from quantum_linear_systems.utils import make_matrix_hermitian
from quantum_linear_systems.utils import vector_uniformity_entropy
# from trackhhl.hamiltonians.simple_hamiltonian import SimpleHamiltonian
# from trackhhl.hamiltonians.simple_hamiltonian import upscale_pow2
# from trackhhl.toy.simple_generator import SimpleDetectorGeometry
# from trackhhl.toy.simple_generator import SimpleGenerator


class ToyModel:
    """A class representing a generic toy problem for linear systems.

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
    """

    def __init__(
        self, name: str, matrix: np.ndarray, vector: np.ndarray, csol: np.ndarray
    ):
        if not isinstance(name, str):
            raise TypeError(f"Name of ToyModel must be of type str not {type(name)}.")
        for param in [matrix, vector, csol]:
            if not isinstance(param, np.ndarray):
                raise TypeError(
                    f"Matrix, vector and csol of ToyModel must be of type np.ndarray not {type(param)}."
                )
        self.name = name
        self.matrix_a = matrix
        self.vector_b = vector
        self.classical_solution = csol.flatten()

        self.normalize_model()

        print("A =", self.matrix_a, "\n")
        print("b =", self.vector_b)

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits involved in the problem."""
        return int(np.log2(self.matrix_a.shape[0]))

    def normalize_model(self) -> None:
        """Normalize the whole problem to valid quantum states."""
        norm_b = np.linalg.norm(self.vector_b)
        self.matrix_a = self.matrix_a / norm_b
        self.classical_solution = self.classical_solution / norm_b
        self.vector_b = self.vector_b / norm_b

    @staticmethod
    def classically_solve(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Solve a linear system using classical methods.

        Parameters:
           mat (numpy.ndarray): Coefficient matrix of the linear system.
           vec (numpy.ndarray): Right-hand side vector of the linear system.

        Returns:
           numpy.ndarray: The solution vector of the linear system.
        """
        return np.linalg.solve(mat, vec)


class Qiskit4QubitExample(ToyModel):
    """Reproduces the qiskit "4-qubit-HHL" example from
    `https://learn.qiskit.org/course/ch-applications/ solving-linear-systems-of-
    equations-using-hhl-and-its-qiskit-implementation#example1`

    matrix_a =   1  -1/3
                -1/3 1
    vector_b = (1,0)
    classical_solution = (1.125, 0.375)
    """

    def __init__(self) -> None:
        name = "Qiskit4QubitExample"
        matrix_a = np.array([[1, -1 / 3], [-1 / 3, 1]])
        vector_b = np.array([1, 0])
        classical_solution = np.array([[1.125], [0.375]])
        super().__init__(
            name=name, matrix=matrix_a, vector=vector_b, csol=classical_solution
        )


class VolterraProblem(ToyModel):
    """Define the Volterra integral equation problem ready for solving.

    Parameters:
        num_qubits (int): The size of the problem, such that the total size will be 2**num_qubits.
    """

    def __init__(self, num_qubits: int) -> None:
        name = f"VolterraProblem(n={num_qubits})"
        # starting with simplified Volterra integral equation x(t) = 1 - I(x(s)ds)0->t
        total_n = 2**num_qubits
        delta_s = 1 / total_n

        alpha = delta_s / 2

        vec = np.ones((total_n, 1))
        mat = self.volterra_a_matrix(size=total_n, alpha=alpha)

        # expanding them to a hermitian form of A --> A_tilde*x=b_tilde
        a_tilde = make_matrix_hermitian(mat)
        b_tilde = expand_b_vector(vec)

        classical_solution = self.classically_solve(mat, vec)

        super().__init__(
            name=name, matrix=a_tilde, vector=b_tilde, csol=classical_solution
        )

    @staticmethod
    def volterra_a_matrix(size: int, alpha: float) -> np.ndarray:
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
    """Define the classiq demo problem.

    (See https://platform.classiq.io/advanced)
    """

    def __init__(self) -> None:
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

        super().__init__(
            name=name, matrix=matrix_a, vector=vector_b, csol=classical_solution
        )


class RandomNQubitProblem(ToyModel):
    """Define a problem consisting of a random MxM matrix and vector.

    Parameters:
        num_qubits (int): Number of qubits, such that the total size will be M=2**problem_size. Important: If the
        matrix is not hermitian and needs to be made hermitian the actual num_qubits will increase by 1 from the input.
    """

    def __init__(self, num_qubits: int) -> None:
        name = f"RandomNQubitProblem(N={num_qubits})"
        matrix_a = np.random.rand(2**num_qubits, 2**num_qubits)
        matrix_a += (
            matrix_a.T
        )  # make matrix symmetric (don't expand here, to keep problem size to N)
        vector_b = np.random.rand(2**num_qubits)

        classical_solution = self.classically_solve(matrix_a, vector_b)

        super().__init__(
            name=name, matrix=matrix_a, vector=vector_b, csol=classical_solution
        )


def integro_differential_a_matrix(
    a_matrix: np.ndarray, time_discretization_steps: int
) -> np.ndarray:
    """Build a matrix of arbitrary size representing the integro-differential toy
    model."""
    delta_t = 1 / time_discretization_steps
    alpha_n = a_matrix.shape[0]
    identity_block = np.identity(alpha_n)
    zero_block = np.zeros((alpha_n, alpha_n))
    off_diagonal_block = -np.identity(alpha_n) - delta_t * a_matrix
    generated_block = []
    for i in range(time_discretization_steps):
        if i == 0:
            generated_block.append(
                [
                    np.block(
                        [identity_block]
                        + [zero_block for _ in range(time_discretization_steps - 1)]
                    )
                ]
            )
        else:
            generated_block.append(
                [
                    np.block(
                        [
                            [zero_block for _ in range(i - 1)]
                            + [off_diagonal_block, identity_block]
                            + [
                                zero_block
                                for _ in range(time_discretization_steps - (i + 1))
                            ]
                        ]
                    )
                ]
            )
    return np.block(generated_block)


class ScalingTestModel(ToyModel):
    """ToyModel to test how algorithms scale with different properties of the input
    matrix and vector.

    Considerations:
        1. if b is not close to uniform state preparation may be costly
        2. matrix A needs to be s-sparce then exp(iAt) can be calculated in time O(log(N) s**2 t)
        3. matrix needs to be well conditioned

    Parameters:
    matrix_size (int): The size of the matrix (number of rows and columns).
    matrix_s (float): The s-sparsity of the matrix (the matrix has at most s non-zero entries in any row or column).
    matrix_well_conditioned (bool): Whether the matrix singular values lie between the reciprocal of its
        condition number and 1.
    vector_uniformity (float): Uniformity of the random vector b. Float between 0 (non-uniform, sampled from normal
        distribution) and 1 (completely uniform).
    max_num_iterations (int): (optional) Maximum number of attempts to find a matrix that is well-conditioned. Defaults
        to 10.
    """

    def __init__(
        self,
        matrix_size: int = 4,
        matrix_s: int = 2,
        matrix_well_conditioned: bool = True,
        vector_uniformity: float = 1.0,
        max_num_iterations: int = 100,
    ):
        vector_b = generate_random_vector(
            size=matrix_size, uniformity_level=vector_uniformity
        )
        entropy_before = vector_uniformity_entropy(vector_b)
        vector_b = expand_b_vector(vector_b)
        assert np.isclose(entropy_before, vector_uniformity_entropy(vector_b))
        # todo: if we make the matrix hermitian by expanding...
        #  what is the effect on the uniformity of b and therefore the algorithm? entropy does not change-> is this
        #   a good measure?

        matrix_a = generate_s_sparse_matrix(
            matrix_size=matrix_size, s_non_zero_entries=matrix_s
        )
        matrix_a = make_matrix_hermitian(matrix_a)
        iterations = 1

        if matrix_well_conditioned:
            # for well conditioned we want the matrix to be close to 1
            threshold = (
                10 * matrix_size
            )  # Note: small condition numbers are much more difficult for large matrices
        else:
            # for ill conditioned we want the matrix to be a lot larger than 1
            threshold = 1000

        while matrix_well_conditioned != is_matrix_well_conditioned(
            matrix_a, threshold
        ):
            matrix_a = generate_s_sparse_matrix(
                matrix_size=matrix_size, s_non_zero_entries=matrix_s
            )
            matrix_a = make_matrix_hermitian(matrix_a)
            iterations += 1
            print(
                f"matrix_condition_number = {np.linalg.cond(matrix_a)}, frob={np.linalg.cond(matrix_a, 'fro')},"
                f" det(a)={np.linalg.det(matrix_a)}"
            )
            if iterations == max_num_iterations:
                raise ValueError(
                    f"Could not generate matrix where matrix_well_conditioned={matrix_well_conditioned} "
                    f"within {iterations} iterations."
                )

        print("Succeeded in finding a matrix.")
        classical_solution = self.classically_solve(matrix_a, vector_b)
        classical_solution = extract_x_from_expanded(classical_solution)

        matrix_condition_number = np.linalg.cond(matrix_a)
        vector_b_entropy = vector_uniformity_entropy(vector_b)
        name = f"TestNxN_n={matrix_size}_s={matrix_s}_c={matrix_condition_number:.1f}_e={vector_b_entropy:.1f}"
        super().__init__(
            name=name, matrix=matrix_a, vector=vector_b, csol=classical_solution
        )


# class HEPTrackReconstruction(ToyModel):
#     """Toymodel for HEP particle tracing by Davide, taken from
#     `https://github.com/dnicotra/TrackHHL`."""
#
#     def __init__(self, num_detectors: int = 3, num_particles: int = 2) -> None:
#         # Davide used 3,2 for "small" and 3,3 for "large"
#         # Generate a test event
#         detector = SimpleDetectorGeometry(
#             list(range(num_detectors)),
#             [10000 for _ in range(num_detectors)],
#             [10000 for _ in range(num_detectors)],
#             [i + 1 for i in range(num_detectors)],
#         )
#         generator = SimpleGenerator(detector, theta_max=np.pi / 3)
#
#         event = generator.generate_event(num_particles)
#
#         # Initialise Hamiltonian
#         epsilon = 1e-5
#         gamma = 2.0
#         delta = 1.0
#
#         ham = SimpleHamiltonian(epsilon, gamma, delta)
#         ham.construct_hamiltonian(event)
#
#         matrix_a = ham.A.todense() / np.linalg.norm(ham.b)
#         vector_b = ham.b / np.linalg.norm(ham.b)
#         matrix_a, vector_b = upscale_pow2(matrix_a, vector_b)
#         csol = np.linalg.solve(matrix_a, vector_b)
#         super().__init__(
#             name=f"HEPSimpleHamiltonianD{num_detectors}P{num_particles}",
#             matrix=matrix_a,
#             vector=vector_b,
#             csol=csol,
#         )


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    T_N = 4
    a = np.random.random((T_N, T_N))

    result = integro_differential_a_matrix(a, T_N)
    print(result)
