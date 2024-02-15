import unittest

import numpy as np

from quantum_linear_systems.quantum_linear_solver import QuantumLinearSolver


class TestQuantumLinearSolver(unittest.TestCase):
    def test_check_matrix_condition_number(self) -> None:
        qls = QuantumLinearSolver()
        # Test case 1: Identity matrix
        matrix_a = np.eye(3)
        result = qls.check_matrix_condition_number(matrix_a)
        self.assertEqual(result, 1.0)

        # Test case 2: Diagonal matrix with positive values
        matrix_a = np.diag([2, 3, 5])
        result = qls.check_matrix_condition_number(matrix_a)
        self.assertEqual(result, 2.5)

    def test_check_matrix_sparsity(self) -> None:
        qls = QuantumLinearSolver()
        # Test case 1: Zero matrix
        matrix_a = np.zeros((3, 3))
        result = qls.check_matrix_sparsity(matrix_a)
        self.assertEqual(result, 1.0)

        # Test case 2: Identity matrix
        matrix_a = np.eye(3)
        result = qls.check_matrix_sparsity(matrix_a)
        self.assertEqual(result, 6 / 9)


if __name__ == "__main__":
    unittest.main()
