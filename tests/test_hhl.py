import unittest
import numpy as np
from linear_solvers import HHL
from qiskit.quantum_info import Statevector

from quantum_linear_systems.toymodels import qiskit_4qubit_example
from quantum_linear_systems.utils import extract_hhl_solution_vector_from_state_vector
from quantum_linear_systems.hhl_classiq_implementation import classiq_hhl_implementation, verification_of_result


class TestQiskit(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix, self.vector, self.solution = qiskit_4qubit_example()
        self.norm_solution = np.transpose(self.solution / np.linalg.norm(self.solution)).flatten()

    def test_4qubit_example(self):
        naive_hhl_solution = HHL().solve(self.matrix, self.vector)
        self.assertAlmostEqual(naive_hhl_solution.euclidean_norm, np.linalg.norm(self.solution))
        naive_state_vector = Statevector(naive_hhl_solution.state).data
        naive_solution = extract_hhl_solution_vector_from_state_vector(hermitian_matrix=self.matrix,
                                                                       state_vector=naive_state_vector)
        self.assertTrue(np.allclose(self.norm_solution, naive_solution))


class TestClassiq(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix, self.vector, self.solution = qiskit_4qubit_example()
        self.norm_solution = np.transpose(self.solution / np.linalg.norm(self.solution)).flatten()

    def test_4qubit_example(self):
        hhl_circuit, _, _, w_min = classiq_hhl_implementation(matrix_a=self.matrix, vector_b=self.vector, precision=4)
        class_sol, q_sol = verification_of_result(hhl_circuit, num_shots=1000, matrix_a=self.matrix,
                                                  vector_b=self.vector, w_min=w_min)
        q_sol /= np.linalg.norm(q_sol)
        print(self.norm_solution)
        print(q_sol)
        self.assertTrue(np.allclose(class_sol, self.solution))
        self.assertTrue(np.allclose(self.norm_solution, q_sol))


if __name__ == '__main__':
    unittest.main()
