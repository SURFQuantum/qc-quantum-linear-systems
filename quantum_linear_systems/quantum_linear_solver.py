"""LinearSolver class."""

from typing import Any
from typing import Tuple

import numpy as np

from quantum_linear_systems.implementations.hhl_classiq_implementation import (
    solve_hhl_classiq,
)
from quantum_linear_systems.implementations.hhl_qiskit_implementation import (
    solve_hhl_qiskit,
)
from quantum_linear_systems.implementations.vqls_qiskit_implementation import (
    solve_vqls_qiskit,
)


class QuantumLinearSolver:
    """Quantum Linear Solver class to solve problems of the shape Ax=b.

    Currently, supports the following solvers:
    * "hhl_qiskit"
    * "hhl_classiq"
    * "vqls_qiskit"
    """

    def __init__(self) -> None:
        """Initialize the QuantumLinearSolver."""
        self.matrix_a: np.ndarray
        self.vector_b: np.ndarray
        self.name: str
        self.method: str
        self.solution: np.ndarray
        self.qasm_circuit: str
        self.circuit_width: int
        self.circuit_depth: int
        self.run_time: float

    def circuit_data(self) -> Tuple[str, int, int]:
        """Return data about the solution circuit."""
        return self.qasm_circuit, self.circuit_depth, self.circuit_width

    def normalize_model(self) -> None:
        """Normalize the whole problem to valid quantum states."""
        norm_b = np.linalg.norm(self.vector_b)
        self.matrix_a = self.matrix_a / norm_b
        self.vector_b = self.vector_b / norm_b

    def solve(
        self,
        matrix_a: np.ndarray,
        vector_b: np.ndarray,
        method: str,
        file_basename: str,
        **kwargs: Any,
    ) -> np.ndarray:
        """Solve the linear system Ax = b using the specified method.

        Args:
            matrix_a (numpy.ndarray): Coefficient matrix of shape (n, n).
            vector_b (numpy.ndarray): Right-hand side vector of shape (n,).
            method (str): The method to use for solving the linear system.
                Options: 'hhl_qiskit', 'hhl_classiq', 'vqls_qiskit'
            file_basename (str, optional) : Name of the problem (for file saving/printing). E.g. 'VolterraProblem'.

        Returns:
            numpy.ndarray: The solution vector x.
        """
        self.matrix_a, self.vector_b, self.name, self.method = (
            matrix_a,
            vector_b,
            file_basename,
            method,
        )
        # currently all implemented solvers share these requirements
        self.check_matrix_square_hermitian(self.matrix_a)

        # self.normalize_model() # this should be in the respective solver method if it is needed

        # solve
        if method == "hhl_qiskit":
            (
                self.solution,
                self.qasm_circuit,
                self.circuit_depth,
                self.circuit_width,
                self.run_time,
            ) = solve_hhl_qiskit(matrix_a=matrix_a, vector_b=vector_b, **kwargs)
        elif method == "hhl_classiq":
            (
                self.solution,
                self.qasm_circuit,
                self.circuit_depth,
                self.circuit_width,
                self.run_time,
            ) = solve_hhl_classiq(matrix_a=matrix_a, vector_b=vector_b, **kwargs)
        elif method == "vqls_qiskit":
            (
                self.solution,
                self.qasm_circuit,
                self.circuit_depth,
                self.circuit_width,
                self.run_time,
            ) = solve_vqls_qiskit(matrix_a=matrix_a, vector_b=vector_b, **kwargs)
        else:
            raise NotImplementedError(f"Unsupported method: {method}")

        return self.solution

    def save_qasm(self) -> None:
        """Save the qasm circuit as a file."""
        with open(
            f"{self.name}_{self.method}.qasm", "w", encoding="utf-8"
        ) as qasm_file:
            qasm_file.write(self.qasm_circuit)

    @staticmethod
    def check_matrix_square_hermitian(matrix_a: np.ndarray) -> None:
        """Check if the coefficient matrix A is square and Hermitian."""
        if matrix_a.shape[0] != matrix_a.shape[1]:
            raise ValueError(
                f"Input matrix A needs to be square, not "
                f"{matrix_a.shape[0]}x{matrix_a.shape[1]}."
            )
        if not np.allclose(matrix_a, np.conj(matrix_a.T)):
            raise ValueError("Input matrix A is not hermitian!")

    @staticmethod
    def check_matrix_condition_number(matrix_a: np.ndarray) -> float:
        """Check the condition number of the matrix A."""
        eigvals = np.linalg.eigvals(matrix_a)
        abs_eigvals = np.abs(eigvals)
        max_eigval = np.max(abs_eigvals)
        min_eigval = np.min(abs_eigvals[np.nonzero(abs_eigvals)])
        return float(max_eigval / min_eigval)

    @staticmethod
    def check_matrix_sparsity(matrix_a: np.ndarray) -> float:
        """Check the sparsity of the matrix A."""
        num_zeros = np.sum(
            matrix_a == 0
        )  # todo: maybe better to loosen sparsity condition to being close to 0
        num_elements = matrix_a.shape[0] * matrix_a.shape[1]
        return float(num_zeros / num_elements)


if __name__ == "__main__":
    # Example usage:
    A = np.array([[2, -1, 0, 1], [-1, 2, -1, 1], [0, -1, 2, 1], [0, -1, 2, 1]])
    b = np.array([1, 0, -1, 0])

    linear_solver = QuantumLinearSolver()
    sol = linear_solver.solve(
        matrix_a=A, vector_b=b, method="hhl_qiskit", file_basename="example"
    )
    print("Solution:", sol)
