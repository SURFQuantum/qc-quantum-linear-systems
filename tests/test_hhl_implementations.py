"""Test HHL implementations."""

import unittest

import numpy as np
import pytest

from quantum_linear_systems.implementations.hhl_classiq_implementation import (
    solve_hhl_classiq,
)
from quantum_linear_systems.implementations.hhl_qiskit_implementation import (
    solve_hhl_qiskit,
)
from quantum_linear_systems.toymodels import Qiskit4QubitExample


class TestQiskitHHL(unittest.TestCase):
    """Test Qiskit HHL implementation."""

    def setUp(self) -> None:
        self.test_model = Qiskit4QubitExample()

    def test_4qubit_example(self) -> None:
        """Test if qiskit can solve the 4qubit example."""
        q_sol, _, _, width, _ = solve_hhl_qiskit(
            self.test_model.matrix_a, self.test_model.vector_b, show_circuit=False
        )
        self.assertEqual(width, 5)
        self.assertTrue(np.allclose(self.test_model.classical_solution, q_sol))


class TestClassiqHHL(unittest.TestCase):
    """Test Classiq HHL implementation."""

    def setUp(self) -> None:
        self.test_model = Qiskit4QubitExample()

    @pytest.mark.requires_auth
    def test_4qubit_example(self) -> None:
        """Test if classiq can solve the 4qubit example."""
        qpe_register = 3
        q_sol, _, _, width, _ = solve_hhl_classiq(
            self.test_model.matrix_a,
            self.test_model.vector_b,
            qpe_register_size=qpe_register,
            show_circuit=False,
        )

        self.assertEqual(width, 2 + qpe_register)
        print(self.test_model.classical_solution, q_sol)
        print(np.linalg.norm(self.test_model.classical_solution), np.linalg.norm(q_sol))
        self.assertTrue(
            np.allclose(self.test_model.classical_solution, q_sol, atol=0.1)
        )


if __name__ == "__main__":
    unittest.main()
