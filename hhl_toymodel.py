import numpy as np
from itertools import product
from classiq.builtin_functions import StatePreparation
from classiq.builtin_functions import Exponentiation, PhaseEstimation
from classiq.builtin_functions.exponentiation import PauliOperator
from classiq.interface.generator.qpe import (
    ExponentiationScaling,
    ExponentiationSpecification,
)
from classiq.builtin_functions import AmplitudeLoading
from classiq.interface.generator.amplitude_loading import AmplitudeLoadingImplementation
from classiq import Model
from classiq.model import Constraints
import matplotlib
import matplotlib.pyplot as plt

from classiq import Executor
from classiq.execution import IBMBackendPreferences


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


def make_matrix_hermitian(matrix):
    """Creates a hermitian version of a NxM :obj:np.matrix A as a  (N+M)x(N+M) block matrix [[0 ,A], [A_dagger, 0]]."""
    shape = matrix.shape
    upper_zero = np.matrix(np.zeros(shape))
    lower_zero = np.matrix(np.zeros(shape=(shape[1], shape[0])))
    matrix_dagger = matrix.getH()
    hermitian_matrix = np.matrix(np.block([[upper_zero, matrix], [matrix_dagger, lower_zero]]))
    # unittest, check if output is hermitian: A == A_dagger
    assert np.array_equal(hermitian_matrix, hermitian_matrix.getH())
    return hermitian_matrix


def expand_b_vector(unexpanded_vector, non_hermitian_matrix):
    """Expand vector according to the expansion of the matrix to make it hermitian b -> (b 0)."""
    shape = non_hermitian_matrix.shape
    lower_zero = np.matrix(np.zeros(shape=(shape[1], 1)))
    return np.block([[unexpanded_vector], [lower_zero]])


def extract_x_from_expanded(expanded_solution_vector, non_hermitian_matrix):
    """The expanded problem returns a vector y=(0 x), this function returns x from input y."""
    shape = non_hermitian_matrix.shape
    return np.array(expanded_solution_vector[shape[0]:])


def extract_hhl_solution_vector_from_state_vector(hermitian_matrix, state_vector):
    """Extract the solution vector x from the full state vector of the HHL problem which also includes 1 aux. qubit and
    multiple work qubits encoding the eigenvalues.
    """
    size_of_hermitian_matrix = hermitian_matrix.shape[1]
    number_of_qubits_in_result = int(np.log2(len(state_vector)))
    binary_rep = "1" + (number_of_qubits_in_result-1) * "0"
    return np.real(state_vector[int(binary_rep, 2):(int(binary_rep, 2) + size_of_hermitian_matrix)])


Paulidict = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
}


# generate all combinations of Pauli strings of size n
def generate_all_pauli_strings(seq, n):
    for s in product(seq, repeat=n):
        yield "".join(s)


# convert a Paulistring of size n to 2**n X 2**n matrix
def pauli_string_2mat(seq):
    myPmat = Paulidict[seq[0]]
    for p in seq[1:]:
        myPmat = np.kron(myPmat, Paulidict[p])
    return myPmat


# Hilbert-Schmidt-Product of two matrices M1, M2
def hilbert_schmidt(M1, M2):
    return (np.dot(M1.conjugate().transpose(), M2)).trace()


# Naive decomposition, running over all HS products for all Pauli strings
def lcu_naive(H):
    assert H.shape[0] == H.shape[1], "matrix is not square"
    assert H.shape[0] != 0, "matrix is of size 0"
    assert H.shape[0] & (H.shape[0] - 1) == 0, "matrix size is not 2**n"

    n = int(np.log2(H.shape[0]))
    myPualiList = list(generate_all_pauli_strings("IZXY", n))

    mylist = []

    for pstr in myPualiList:
        co = (1 / 2**n) * hilbert_schmidt(pauli_string_2mat(pstr), H)
        if co != 0:
            mylist = mylist + [(pstr, co)]

    return mylist


def state_preparation(vector_b, sp_upper):
    vector_b = tuple(vector_b)
    # sp_upper = precision of the State Preparation
    return StatePreparation(
        amplitudes=vector_b, error_metric={"L2": {"upper_bound": sp_upper}})


def quantum_phase_estimation(precision):
    po = PauliOperator(pauli_list=paulis)
    exp_params = Exponentiation(
        pauli_operator=po,
        evolution_coefficient=-2 * np.pi,
    )

    return PhaseEstimation(
        size=precision,
        unitary_params=exp_params,
        exponentiation_specification=ExponentiationSpecification(
            scaling=ExponentiationScaling(max_depth=100, max_depth_scaling_factor=2)
        ),
    )


def verification_of_result(circuit, num_shots, matrix_a, vector_b):
    res_hhl = Executor(
        backend_preferences=IBMBackendPreferences(backend_name="aer_simulator_statevector"),
        num_shots=num_shots,
    ).execute(circuit)

    total_q = circuit.data.width  # total number of qubits of the whole circuit

    target_pos = (
            total_q - 1 - res_hhl.output_qubits_map["target"][0]
    )  # position of control qubit (corrected for endianness)
    sol_pos = [
        total_q - 1 - qn for qn in res_hhl.output_qubits_map["solution"]
    ]  # position of solution (corrected for endianness)

    canonical_list = np.array(list("0" * (total_q)))  # we start with a string of zeros
    canonical_list[
        target_pos
    ] = "1"  # we are interested in strings having 1 on their target qubit

    qsol = list()
    for i in range(2 ** len(sol_pos)):
        templist = canonical_list.copy()
        templist[sol_pos] = list(np.binary_repr(i, len(sol_pos))[::-1])
        qsol.append(np.round(complex(res_hhl.state_vector["".join(templist)]) / w_min, 5))

    print("first", qsol)
    sol_classical = np.linalg.solve(matrix_a, vector_b)
    global_phase = np.angle(qsol)
    qsol_corrected = np.real(qsol / np.exp(1j * global_phase))
    print("classical:  ", sol_classical)
    print("HHL:        ", qsol_corrected)
    print(
        "relative distance:  ",
        round(
            np.linalg.norm(sol_classical - qsol_corrected)
            / np.linalg.norm(sol_classical)
            * 100,
            1,
        ),
        "%",
    )
    return sol_classical, qsol_corrected


def define_volterra_problem(n):
    # starting with simplified Volterra integral equation x(t) = 1 - I(x(s)ds)0->t
    N = 2 ** n
    delta_s = 1 / N

    alpha = delta_s / 2

    vec = np.ones((N, 1))

    # prepare matrix A and vector b to be used for HHL, expanding them to a hermitian form of A --> A_tilde*x=b_tilde
    mat = volterra_a_matrix(size=N, a=alpha)

    print("A =", mat, "\n")
    print("b =", vec)

    # expand
    a_tilde = make_matrix_hermitian(mat)
    b_tilde = expand_b_vector(vec, mat)

    print("A_tilde =", a_tilde, "\n")
    print("b_tilde =", b_tilde)

    return a_tilde, b_tilde


def define_demo_problem():
    mat = np.array(
        [
            [0.28, -0.01, 0.02, -0.1],
            [-0.01, 0.5, -0.22, -0.07],
            [0.02, -0.22, 0.43, -0.05],
            [-0.1, -0.07, -0.05, 0.42],
        ]
    )

    vec = np.array([1, 2, 4, 3])
    vec = vec / np.linalg.norm(vec)

    print("A =", mat, "\n")
    print("b =", vec)
    return mat, vec


def verify_matrix_sym_and_pos_ev(matrix):
    if not np.allclose(A, A.T, rtol=1e-6, atol=1e-6):
        raise Exception("The matrix is not symmetric")
    w, v = np.linalg.eig(A)
    for lam in w:
        if lam < 0 or lam > 1:
            raise NotImplementedError(f"Eigenvalues are not in (0,1), lam_min={min(w)}")


if __name__ == "__main__":
    # input params
    n = 1
    precision = 4

    A, b = define_volterra_problem(n)
    # A, b = define_demo_problem()

    # verifying that the matrix is symmetric and have eigenvalues in [0,1)
    verify_matrix_sym_and_pos_ev(matrix=A)

    paulis = lcu_naive(A)
    # print("Pauli strings list: \n")
    # for p in paulis:
    #     print(p[0], ": ", np.round(p[1], 3))

    print("\n Number of qubits for matrix representation =", len(paulis[0][0]))

    # Step 1: state preparation
    if np.linalg.norm(b) != 1:
        print(f"Normalizing A and b by {np.linalg.norm(b)}")
    b_normalized = b / np.linalg.norm(b)
    A_normalized = A / np.linalg.norm(b)
    sp = state_preparation(vector_b=b_normalized, sp_upper=0.00)

    # Step 2 : Quantum Phase Estimation
    qpe = quantum_phase_estimation(precision=4)

    # Step 3 : Eigenvalue Inversion

    w_min = (
            1 / 2 ** precision
    )  # for qpe register of size m, this is the minimal value which can be encoded
    expression = f"{w_min}/(x)"
    al_params = AmplitudeLoading(
        size=precision,
        expression=expression,
        implementation=AmplitudeLoadingImplementation.GRAYCODE,
    )

    # Step 4 Inverse QPE

    model_hhl = Model()
    sp_out = model_hhl.StatePreparation(params=sp)
    qpe_out = model_hhl.PhaseEstimation(params=qpe, in_wires={"IN": sp_out["OUT"]})
    al_out = model_hhl.AmplitudeLoading(
        params=al_params,
        in_wires={"AMPLITUDE": qpe_out["PHASE_ESTIMATION"]},
    )
    i_qpe_out = model_hhl.PhaseEstimation(
        params=qpe,
        is_inverse=True,
        release_by_inverse=True,
        in_wires={
            "PHASE_ESTIMATION": al_out["AMPLITUDE"],
            "OUT": qpe_out["OUT"],
        },
    )

    model_hhl.set_outputs({"target": al_out["TARGET"], "solution": i_qpe_out["IN"]})

    # Synth circuit

    circuit_hhl = model_hhl.synthesize()
    # circuit_hhl = model_hhl.synthesize(constraints=Constraints(max_width=10))
    circuit_hhl.show_interactive()
    print("depth = ", circuit_hhl.transpiled_circuit.depth)

    # verify against classical solution
    csol, qsol = verification_of_result(circuit=circuit_hhl, num_shots=1, matrix_a=A_normalized, vector_b=b_normalized)

    matplotlib.use('Qt5Agg')
    plt.plot(csol, "bo", label="classical")
    plt.plot(qsol, "ro", label="HHL")
    plt.legend()
    plt.xlabel("$i$")
    plt.ylabel("$x_i$")
    plt.show()

    if np.linalg.norm(csol - qsol) / np.linalg.norm(csol) > 0.2:
        raise RuntimeError("The HHL solution is too far from the classical one, please verify your algorithm.")