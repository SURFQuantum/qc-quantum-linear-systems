"""Test VQLS implementations."""
import unittest
import numpy as np
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes

from quantum_linear_systems.toymodels import Qiskit4QubitExample
from quantum_linear_systems.implementations.vqls_qiskit_implementation import solve_vqls_qiskit, postprocess_solution


class TestQiskitVQLS(unittest.TestCase):
    """Test Qiskit HHL implementation."""
    def setUp(self) -> None:
        self.test_model = Qiskit4QubitExample()
        self.test_ansatz = RealAmplitudes(num_qubits=1, entanglement="full", reps=3, insert_barriers=False)

    def test_4qubit_example(self):
        """Test if qiskit can solve the 4qubit example."""
        q_sol, _, _, width, _ = solve_vqls_qiskit(self.test_model.matrix_a, self.test_model.vector_b,
                                                  ansatz=self.test_ansatz, show_circuit=False)
        self.assertEqual(width, 1)
        self.assertTrue(np.allclose(self.test_model.classical_solution, q_sol, atol=1e-3))

    def test_postprocessing(self):
        """Test the functionality of the postprocessing function."""
        # test normalization
        matrix_a = np.random.rand(2, 2)
        vector_b = np.random.rand(2)
        true_solution = np.linalg.solve(matrix_a, vector_b)
        # vqls outputs normalized solution
        sol_vqls = true_solution / np.linalg.norm(true_solution)
        self.assertFalse(np.allclose(true_solution, sol_vqls))
        post_sol = postprocess_solution(matrix_a, vector_b, sol_vqls)
        self.assertTrue(np.allclose(true_solution, post_sol))

        # test sign flipping
        vqls_sol_flipped = np.linalg.solve(matrix_a, vector_b)
        vqls_sol_flipped = - vqls_sol_flipped
        vqls_sol_flipped = vqls_sol_flipped / np.linalg.norm(vqls_sol_flipped)

        self.assertFalse(np.allclose(true_solution, vqls_sol_flipped))
        post_sol = postprocess_solution(matrix_a, vector_b, vqls_sol_flipped)
        self.assertTrue(np.allclose(true_solution, post_sol))
