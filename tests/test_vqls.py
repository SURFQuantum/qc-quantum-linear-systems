"""Test VQLS implementations."""
import unittest
import numpy as np
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes

from quantum_linear_systems.toymodels import Qiskit4QubitExample
from quantum_linear_systems.vqls_qiskit_implementation import qiskit_vqls


class TestQiskitVQLS(unittest.TestCase):
    """Test Qiskit HHL implementation."""
    def setUp(self) -> None:
        self.test_model = Qiskit4QubitExample()
        self.test_ansatz = RealAmplitudes(num_qubits=1, entanglement="full", reps=3, insert_barriers=False)

    def test_4qubit_example(self):
        """Test if qiskit can solve the 4qubit example."""
        q_sol, csol, _, width, _ = qiskit_vqls(model=self.test_model, ansatz=self.test_ansatz, show_circuit=False)
        self.assertEqual(width, 1)
        self.assertTrue(np.allclose(csol, q_sol, atol=1e-3))
