"""Implementations of toy-models that can be imported by the algorithms."""
import numpy as np


def volterra_a_matrix(size, a):
    """Creates a matrix representing the linear system of the Volterra integral equation x(t) = 1 - INT(x(s)ds).
    Parameters
    ----------
    size : int
        size of the square matrix to be created.
    a : float
        alpha = delta S / 2 parametrization.
    Returns
    -------
    a_matrix : np.matrix
        size x size square matrix.
    """
    matrix_a = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                # diagonal first entry 1, rest (1 + a)
                if i == 0:
                    matrix_a[i, j] = 1
                else:
                    matrix_a[i, j] = 1 + a
            elif j == 0:
                # first column (except first entry which is covered above) is all a
                matrix_a[i, j] = a
            elif 0 < j < i:
                # rest of lower bottom triangle is 2a
                matrix_a[i, j] = 2 * a
    return np.matrix(matrix_a)
