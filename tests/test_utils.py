import unittest
import numpy as np
from quantum_linear_systems.utils import make_matrix_hermitian, expand_b_vector, extract_x_from_expanded, \
    extract_hhl_solution_vector_from_state_vector


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    def setUp(self):
        """Set up some matrices and vectors for testing."""
        self.matrix = np.array([[1, 2], [3, 4]])
        self.vector = np.array([[5], [6]])

    def test_make_matrix_hermitian(self):
        hermitian = make_matrix_hermitian(self.matrix)
        self.assertTrue(np.array_equal(hermitian, hermitian.conj().T))

    def test_expand_b_vector(self):
        expanded_vector = expand_b_vector(self.vector, self.matrix)
        self.assertEqual(expanded_vector.shape, (4, 1))

    def test_extract_x_from_expanded(self):
        expanded_vector = expand_b_vector(self.vector, self.matrix)
        extracted_vector = extract_x_from_expanded(expanded_vector, self.matrix)
        self.assertEqual(extracted_vector.shape, (2, 1))

    def test_extract_hhl_solution_vector_from_state_vector(self):
        hermitian = make_matrix_hermitian(self.matrix)
        state_vector = np.array([0, 0, 1, 0, 0, 0, 0, 0])  # 2**3 = 8 for a 2x2 matrix
        solution_vector = extract_hhl_solution_vector_from_state_vector(hermitian, state_vector)
        self.assertEqual(solution_vector.shape, (4,))  # Checking shape of the output
        # todo: this only checks the shape, nothing else. This should be improved.


if __name__ == '__main__':
    unittest.main()
