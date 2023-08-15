"""Test HHL implementations."""
import unittest
import numpy as np

from quantum_linear_systems.toymodels import Qiskit4QubitExample
from quantum_linear_systems.hhl_classiq_implementation import classiq_hhl
from quantum_linear_systems.hhl_qiskit_implementation import qiskit_hhl


class TestQiskitHHL(unittest.TestCase):
    """Test Qiskit HHL implementation."""
    def setUp(self) -> None:
        self.test_model = Qiskit4QubitExample()

    def test_4qubit_example(self):
        """Test if qiskit can solve the 4qubit example."""
        q_sol, csol, _, width, _ = qiskit_hhl(model=self.test_model,
                                              show_circuit=False)
        self.assertEqual(width, 5)
        self.assertTrue(np.allclose(csol, q_sol))


class TestClassiqHHL(unittest.TestCase):
    """Test Classiq HHL implementation."""
    def setUp(self) -> None:
        self.test_model = Qiskit4QubitExample()

    def test_4qubit_example(self):
        """Test if classiq can solve the 4qubit example."""
        qpe_register = 3
        q_sol, csol, _, width, _ = classiq_hhl(model=self.test_model, qpe_register_size=qpe_register,
                                               show_circuit=False)

        q_sol /= np.linalg.norm(q_sol)
        self.assertEqual(width, 2 + qpe_register)
        self.assertTrue(np.allclose(csol, q_sol, atol=.1))


if __name__ == '__main__':
    unittest.main()
