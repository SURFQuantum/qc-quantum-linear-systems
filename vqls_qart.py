import numpy as np


def integro_differential_a(a_matrix, t_n):
    """Build a matrix of arbitrary size representing the integro-differential toy model."""
    delta_t = 1 / t_n
    alpha_n = a_matrix.shape[0]
    identity_block = np.identity(alpha_n)
    zero_block = np.zeros((alpha_n, alpha_n))
    off_diagonal_block = - np.identity(alpha_n) - delta_t * a_matrix
    generated_block = []
    for i in range(t_n):
        if i == 0:
            generated_block.append([np.block([identity_block] + [zero_block for _ in range(t_n - 1)])])
        else:
            generated_block.append([np.block([[zero_block for _ in range(i-1)] +
                                             [off_diagonal_block, identity_block] +
                                             [zero_block for _ in range(t_n - (i + 1))]])])
    return np.block(generated_block)


def decompose_into_unitaries(matrix):
    return


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    t_n = 4
    a = np.random.random((t_n, t_n))

    result = integro_differential_a(a, t_n)
    print(result)
