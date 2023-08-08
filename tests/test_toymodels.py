import unittest
import numpy as np

from quantum_linear_systems.toymodels import qiskit_4qubit_example, volterra_problem, classiq_demo_problem


class TestToymodels(unittest.TestCase):
    def test_output_type(self):
        for toymodel in [qiskit_4qubit_example(), volterra_problem(1), classiq_demo_problem()]:
            self.assertTrue(len(toymodel) == 4)
            self.assertTrue(type(toymodel[0]) == np.ndarray)
            self.assertTrue(type(toymodel[1]) == np.ndarray)
            self.assertTrue(type(toymodel[2]) == np.ndarray)
            self.assertTrue(type(toymodel[3]) == str)
