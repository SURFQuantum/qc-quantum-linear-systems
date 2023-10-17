"""Test utility functions."""
import unittest

import numpy as np

from quantum_linear_systems.utils import expand_b_vector
from quantum_linear_systems.utils import extract_hhl_solution_vector_from_state_vector
from quantum_linear_systems.utils import extract_x_from_expanded
from quantum_linear_systems.utils import make_matrix_hermitian


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    def setUp(self):
        """Set up some matrices and vectors for testing."""
        self.matrix = np.array([[1, 2, 3], [4, 5, 6]])
        self.square_matrix = np.array([[1, 2], [4, 5]])
        self.vector = np.array([[7], [8]])
        self.test_matrices = [self.matrix, self.square_matrix]

    def test_make_matrix_hermitian(self):
        """Test whether the matrix is correctly made hermitian."""
        for mat in self.test_matrices:
            hermitian = make_matrix_hermitian(mat)
            self.assertTrue(np.array_equal(hermitian, hermitian.conj().T))

    def test_expand_b_vector(self):
        """Test whether vector b is correctly expanded."""
        for mat in self.test_matrices:
            expanded_vector = expand_b_vector(self.vector, non_square_matrix=mat)
            self.assertEqual(expanded_vector.shape, (mat.shape[0] + mat.shape[1],))
            self.assertTrue(np.array_equal(expanded_vector, np.array([[7], [8], *(mat.shape[1] * [[0]])]).flatten()))

    def test_extract_x_from_expanded(self):
        """Test whether x is correctly extracted from an expanded vector (0 x)."""
        expanded_vector = expand_b_vector(self.vector)
        extracted_vector = extract_x_from_expanded(expanded_vector)
        self.assertTrue(np.array_equal(extracted_vector, np.zeros(len(self.vector))))
        extracted_vector = extract_x_from_expanded(np.array([0, 0, 1, 1]))
        self.assertTrue(np.array_equal(extracted_vector, np.array([1, 1])))
        expanded_vector = expand_b_vector(self.vector)
        extracted_vector = extract_x_from_expanded(expanded_vector)
        self.assertTrue(np.array_equal(extracted_vector, np.zeros(len(self.vector))))

    def test_extract_hhl_solution_vector_from_state_vector(self):
        """Test whether the solution vector is correctly extracted."""
        hermitian = make_matrix_hermitian(self.matrix)
        state_vector = np.array([0, 0, 1, 0, 0, 0, 0, 0])  # 2**3 = 8 for a 2x2 matrix
        solution_vector = extract_hhl_solution_vector_from_state_vector(hermitian, state_vector)
        self.assertEqual(solution_vector.shape, (4,))  # Checking shape of the output


if __name__ == '__main__':
    unittest.main()
