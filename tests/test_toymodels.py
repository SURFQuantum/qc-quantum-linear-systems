"""Test ToyModel implementations."""
import unittest

from quantum_linear_systems.toymodels import Qiskit4QubitExample, VolterraProblem, ClassiqDemoExample


class TestToyModels(unittest.TestCase):
    """Unittest the ToyModels."""
    def test_init(self):
        """Test correct initialization."""
        model1 = Qiskit4QubitExample(2)
        self.assertEqual(model1.problem_size, 1)
        model2 = VolterraProblem(2)
        self.assertEqual(model2.problem_size, 2)
        model3 = ClassiqDemoExample(2)
        self.assertEqual(model3.problem_size, 2)
