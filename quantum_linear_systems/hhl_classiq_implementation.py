"""HHL implementation using Classiq."""
import time
from itertools import product

import numpy as np

from classiq.builtin_functions import StatePreparation, Exponentiation, PhaseEstimation, AmplitudeLoading
from classiq.builtin_functions.exponentiation import PauliOperator
from classiq.interface.generator.qpe import (
    ExponentiationScaling,
    ExponentiationSpecification,
)
from classiq.interface.generator.amplitude_loading import AmplitudeLoadingImplementation
from classiq import Model, execute, synthesize, show, GeneratedCircuit
from classiq.execution import ExecutionPreferences, IBMBackendPreferences
from classiq.synthesis import set_execution_preferences
from classiq.execution import ExecutionDetails

from quantum_linear_systems.toymodels import ToyModel, ClassiqDemoExample
from quantum_linear_systems.utils import (extract_x_from_expanded, print_results,
                                          relative_distance_quantum_classical_solution)


Paulidict = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
}


# generate all combinations of Pauli strings of size n
def generate_all_pauli_strings(seq, size):
    """
    Generate all combinations of Pauli strings of size n.

    Parameters:
        seq (str): The string of Pauli operators (I, Z, X, Y).
        size (int): The size of the output Pauli strings.

    Returns:
        A generator of all combinations of Pauli strings of size n.
    """
    for string in product(seq, repeat=size):
        yield "".join(string)


# convert a Paulistring of size n to 2**n X 2**n matrix
def pauli_string_2mat(seq):
    """
    Convert a Pauli string of size n to a 2**n x 2**n matrix.

    Parameters:
        seq (str): The string of Pauli operators (I, Z, X, Y).

    Returns:
        A 2**n x 2**n matrix representation of the Pauli string.
    """
    p_matrix = Paulidict[seq[0]]
    for p_string in seq[1:]:
        p_matrix = np.kron(p_matrix, Paulidict[p_string])
    return p_matrix


# Hilbert-Schmidt-Product of two matrices M1, M2
def hilbert_schmidt(m_1, m_2):
    """
    Compute the Hilbert-Schmidt-Product of two matrices M1, M2.

    Parameters:
        m_1 (np.ndarray): The first matrix.
        m_2 (np.ndarray): The second matrix.

    Returns:
        The Hilbert-Schmidt inner product of the two matrices.
    """
    return (np.dot(m_1.conjugate().transpose(), m_2)).trace()


# Naive decomposition, running over all HS products for all Pauli strings
def lcu_naive(herm_mat):
    """
    Naive LCU (linear combination of unitary operations) decomposition, running over all HS products for all Pauli
    strings.

    Parameters:
        herm_mat (np.ndarray): The input Hermitian matrix.

    Returns:
        A list of tuples, each containing a Pauli string and the corresponding coefficient.
    """
    assert herm_mat.shape[0] == herm_mat.shape[1], "matrix is not square"
    assert herm_mat.shape[0] != 0, "matrix is of size 0"
    assert herm_mat.shape[0] & (herm_mat.shape[0] - 1) == 0, "matrix size is not 2**n"

    num_qubits = int(np.log2(herm_mat.shape[0]))
    pauli_strings = list(generate_all_pauli_strings("IZXY", num_qubits))

    mylist = []

    for pstr in pauli_strings:
        coeff = (1 / 2**num_qubits) * hilbert_schmidt(pauli_string_2mat(pstr), herm_mat)
        if coeff != 0:
            mylist = mylist + [(pstr, coeff)]

    return mylist


def state_preparation(vector_b: np.ndarray, sp_upper: float) -> StatePreparation:
    """
    Prepare the state based on the input vector.

    Parameters:
        vector_b (np.ndarray): The input vector.
        sp_upper (float): The upper bound of the L2 error metric for the state preparation.

    Returns:
        A StatePreparation object with the desired amplitudes and error metric.
    """
    vector_b = tuple(vector_b)
    # sp_upper = precision of the State Preparation
    return StatePreparation(
        amplitudes=vector_b, error_metric={"L2": {"upper_bound": sp_upper}})


def quantum_phase_estimation(paulis: list, qpe_register_size: int) -> PhaseEstimation:
    """
    Perform Quantum Phase Estimation (QPE) with the specified precision.

    Parameters:
        paulis (list) : List of pauli matrices.
        qpe_register_size (int): The desired size of the QPE register.

    Returns:
        A PhaseEstimation object configured with the specified precision.
    """
    exp_params = Exponentiation(
        pauli_operator=PauliOperator(pauli_list=paulis),
        evolution_coefficient=-2 * np.pi,
    )

    return PhaseEstimation(
        size=qpe_register_size,
        unitary_params=exp_params,
        exponentiation_specification=ExponentiationSpecification(
            scaling=ExponentiationScaling(
                max_depth=250,
                max_depth_scaling_factor=2
            )
        ),
    )


def verification_of_result(circuit, w_min, sol_classical):
    """
    Verify the result of the quantum algorithm by comparing with the classical solution.

    Parameters:
        circuit (str): The quantum circuit.
        w_min
        sol_classical (np.ndarray) : classical solution vector

    Returns:
        A tuple containing the classical solution and the solution obtained from the quantum algorithm.
    """
    results = execute(circuit)
    res_hhl = ExecutionDetails.parse_obj(results[0].value)
    circuit = GeneratedCircuit.parse_raw(circuit)

    # qsol_pure= Statevector(circuit).data
    # qsol_pure = [0] * 4

    total_q = circuit.data.width  # total number of qubits of the whole circuit

    target_pos = res_hhl.physical_qubits_map["target"][0]  # position of control qubit

    sol_pos = list(res_hhl.physical_qubits_map["solution"])  # position of solution

    canonical_list = np.array(list("0" * total_q))  # we start with a string of zeros
    canonical_list[
        target_pos
    ] = "1"  # we are interested in strings having 1 on their target qubit

    quantum_solution = []
    for i in range(2 ** len(sol_pos)):
        templist = canonical_list.copy()
        templist[sol_pos] = list(np.binary_repr(i, len(sol_pos)))
        quantum_solution.append(np.round(complex(res_hhl.state_vector["".join(templist)]) / w_min, 5))

    if len(quantum_solution) > len(sol_classical):
        # extract the solution from the extended vector
        quantum_solution = extract_x_from_expanded(np.array(quantum_solution))

    print("first", quantum_solution)
    global_phase = np.angle(quantum_solution)
    qsol_corrected = np.real(quantum_solution / np.exp(1j * global_phase))
    print("classical:  ", sol_classical)
    print("HHL:        ", qsol_corrected)
    print(
        "relative distance:  ",
        round(
            relative_distance_quantum_classical_solution(quantum_solution=qsol_corrected,
                                                         classical_solution=sol_classical), 1,
        ),
        "%",
    )
    return qsol_corrected


def verify_matrix_sym_and_pos_ev(mat):
    """
    Verify that the input matrix is symmetric and has positive eigenvalues.

    Parameters:
        mat (np.ndarray): The input matrix.
    """
    if not np.allclose(mat, mat.T, rtol=1e-6, atol=1e-6):
        raise ValueError("The matrix is not symmetric")
    eigenvalues, _ = np.linalg.eig(mat)
    for lam in eigenvalues:
        if lam < 0:
            raise ValueError("The matrix has negative eigenvalues")
        # the original classiq workshop checked if the EV are in [0,1)
        if lam > 1:
            print("The matrix has eigenvalues larger than 1: ", lam)


def classiq_hhl_implementation(matrix_a: np.ndarray, vector_b: np.ndarray, qpe_register_size: int = None):
    """Classiq HHL implementation based on https://docs.classiq.io/latest/tutorials/advanced/hhl/ ."""
    # verifying that the matrix is symmetric and hs eigenvalues in [0,1)
    # verify_matrix_sym_and_pos_ev(mat=matrix_a)

    if np.linalg.norm(vector_b) != 1:
        print(f"Normalizing A and b by {np.linalg.norm(vector_b)}")

    matrix_a = matrix_a / np.linalg.norm(vector_b)
    vector_b = vector_b / np.linalg.norm(vector_b)

    solution_register_size = int(np.log2(len(vector_b)))
    if qpe_register_size is None:
        # calculate size of qpe_register from matrix
        kappa = np.linalg.cond(matrix_a)    # condition number of matrix
        neg_vals = True     # whether matrix has negative eigenvalues
        qpe_register_size = max(solution_register_size + 1, int(np.ceil(np.log2(kappa + 1)))) + neg_vals
    print(f"Size of solution register is {solution_register_size} , QPE registers is {qpe_register_size}.")

    model_hhl = Model()
    # Step 1: state preparation
    sp_out = model_hhl.StatePreparation(params=state_preparation(vector_b=vector_b, sp_upper=1e-2/3))
    # Note: value of sp_upper: qiskit hhl.py line 101: epsilon=1e-2, line 120 state prep.: epsilon_s = epsilon / 3

    # Step 2 : Quantum Phase Estimation
    qpe = quantum_phase_estimation(paulis=lcu_naive(matrix_a), qpe_register_size=qpe_register_size)
    qpe_out = model_hhl.PhaseEstimation(params=qpe, in_wires={"IN": sp_out["OUT"]})

    # Step 3 : Eigenvalue Inversion
    w_min = 1 / 2 ** qpe_register_size  # for qpe register of size m, this is the minimal value which can be encoded
    al_out = model_hhl.AmplitudeLoading(
        params=AmplitudeLoading(
            size=qpe_register_size,
            expression=f"{w_min}/(x)",
            implementation=AmplitudeLoadingImplementation.GRAYCODE,
        ),
        in_wires={"AMPLITUDE": qpe_out["PHASE_ESTIMATION"]},
    )

    # Step 4 Inverse QPE
    i_qpe_out = model_hhl.PhaseEstimation(
        params=qpe,
        is_inverse=True,
        release_by_inverse=True,
        in_wires={
            "PHASE_ESTIMATION": al_out["AMPLITUDE"],
            "OUT": qpe_out["OUT"],
        },
    )

    model_hhl.sample()

    model_hhl.set_outputs({"target": al_out["TARGET"], "solution": i_qpe_out["IN"]})

    # set Execution Preferences
    serialized_hhl_model = model_hhl.get_model()
    serialized_hhl_model = set_execution_preferences(
        serialized_hhl_model,
        execution_preferences=ExecutionPreferences(
            num_shots=1, backend_preferences=IBMBackendPreferences(
                backend_service_provider="IBM Quantum", backend_name="aer_simulator_statevector"
            )
        ),
    )

    # Synth circuit
    qprog_hhl = synthesize(serialized_hhl_model)

    return qprog_hhl, matrix_a, vector_b, w_min


def classiq_hhl(model: ToyModel, qpe_register_size: int = None, show_circuit: bool = True, save_qasm=False):
    """Full implementation unified between classiq and qiskit."""
    start_time = time.time()

    circuit_hhl, _, _, w_min = classiq_hhl_implementation(matrix_a=model.matrix_a,
                                                          vector_b=model.vector_b,
                                                          qpe_register_size=qpe_register_size)
    if show_circuit:
        show(circuit_hhl)

    gen_circ = GeneratedCircuit.parse_raw(circuit_hhl)
    circuit_depth = gen_circ.transpiled_circuit.depth
    if save_qasm:
        qasm_content = gen_circ.transpiled_circuit.qasm
        with open(f"{model.name}_classiq_hhl.qasm", "w") as qasm_file:
            qasm_file.write(qasm_content)
    circuit_width = len(gen_circ.analyzer_data.qubits)
    print("depth = ", circuit_depth)
    print("width = ", circuit_width)

    # verify against classical solution
    quantum_solution = verification_of_result(circuit=circuit_hhl, w_min=w_min, sol_classical=model.classical_solution)
    # normalize
    quantum_solution /= np.linalg.norm(quantum_solution)

    return quantum_solution, model.classical_solution, circuit_depth, circuit_width, time.time() - start_time


if __name__ == "__main__":
    # input params
    n: int = 2

    toymodel = ClassiqDemoExample()

    qsol, csol, depth, width, run_time = classiq_hhl(model=toymodel, show_circuit=True, save_qasm=True)

    print_results(quantum_solution=qsol, classical_solution=csol, run_time=run_time, name=toymodel.name, plot=True)
