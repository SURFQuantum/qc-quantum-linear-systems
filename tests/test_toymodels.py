"""Test ToyModel implementations."""
import unittest

from quantum_linear_systems.toymodels import Qiskit4QubitExample, VolterraProblem, ClassiqDemoExample, ScalingTestModel


class TestToyModels(unittest.TestCase):
    """Unittest the ToyModels."""
    def test_default_values(self):
        """Test correct initialization."""
        model1 = Qiskit4QubitExample()
        self.assertEqual(model1.num_qubits, 1)
        model2 = VolterraProblem(2)
        self.assertEqual(model2.num_qubits, 2+1)  # volterra needs to be made hermitian so num_qubits grows by 1
        model3 = ClassiqDemoExample()
        self.assertEqual(model3.num_qubits, 2)
        model4 = ScalingTestModel()
        self.assertEqual(model4.num_qubits, 3)
